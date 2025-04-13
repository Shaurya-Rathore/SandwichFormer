import torch
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from datasets import load_dataset
from pathlib import Path
from utils.dataset import CoLADataset
from utils.model import build_transformer
from utils.config import get_weights_path
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR
import wandb
from sklearn.metrics import confusion_matrix, f1_score

##get wandb here

wandb.init(
    # set the wandb project where this run will be logged
    project="SandwichFormer - CoLA",

    # track hyperparameters and run metadata
    config={
    "architecture": "BERT(Restructured)",
    "dataset": "CoLA",
    "epochs": 6,
    }
)

def get_all_sentences():
    ds = load_dataset("glue", "cola", split='train')
    for item in ds:
        yield item["sentence"]

    
def get_or_build_tokenizer(config):
    tokenizer_path = Path(config['tokenizer_file'])
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"], min_frequency = 2)
        tokenizer.train_from_iterator(get_all_sentences(), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config, device):
    train_data_raw = load_dataset("glue", "cola", split='train')
    val_data_raw = load_dataset("glue", "cola", split='validation')
    test_data_raw = load_dataset("glue","cola", split='test')
    tokenizer = get_or_build_tokenizer(config)

    train_ds = CoLADataset(train_data_raw, tokenizer, config['seq_len'])
    val_ds = CoLADataset(val_data_raw, tokenizer, config['seq_len'])
    test_ds = CoLADataset(test_data_raw, tokenizer, config['seq_len'])
    
    max_len = 0
    
    for item in train_data_raw:
        src_id = tokenizer.encode(item['sentence']).ids
        max_len = max(max_len, len(src_id))
    
    print(f"Max Length for sentence = {max_len}")
        
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle = True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle = True)
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle= True)
    return train_dataloader, val_dataloader, tokenizer, test_dataloader

def validate(model, dataloader, device):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    output_all = []
    labels_all = []
    
    with torch.no_grad():

        for batch in dataloader:
            input_ids = batch["encoder_input"].to(device)
            attention_mask = batch["encoder_mask"].to(device)
            label = batch["label"].to(device)
            outputs = model.encode(input_ids, attention_mask)
            classification_output = model.project(outputs)
            classification_output = classification_output.transpose(0, 1)
            classification_output = torch.squeeze(classification_output, dim=0)

            output = threshold(classification_output, 0.5)
            output_np = output.cpu().detach().numpy()
            label_np = label.cpu().detach().numpy()
            output_all.extend(output_np)
            labels_all.extend(label_np)
            correct_predictions += (output == label).sum().item()
            total_predictions += label.size(0)
        
        print(f1_score(labels_all,output_all))
        print(confusion_matrix(labels_all,output_all))
            
    model.train()
    return correct_predictions / total_predictions * 100.0

def get_model(config, inp_vocab_size,device):
    model = build_transformer(
        inp_vocab_size, 
        config['num_labels'], 
        config['seq_len'], 
        config['d_model'], 
        config['num_layers'], 
        config['num_heads'], 
        config['dropout'], 
        config['intermediate_size'],
        device
        )
    return model

def threshold(tensor, threshold_value):
    # Convert values above threshold to 1 and below or equal to threshold to 0
    return torch.where(tensor > threshold_value, torch.tensor(1), torch.tensor(0))


def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    train_dataloader, val_dataloader, tokenizer, test_dataloader = get_ds(config, device)
    model = get_model(config, tokenizer.get_vocab_size(),device)
    model.to(device)
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps= 1e-9)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    
    init_epoch =0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        init_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        
    loss_fn = nn.BCELoss().to(device)
    best_val_accuracy = 0
    
    for epoch in range(init_epoch, config['epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        per_epoch_outputs = []
        per_epoch_labels = []
        for batch in batch_iterator:
            optimizer.zero_grad(set_to_none=True)
            
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            label = batch['label'].to(device)
            
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, Seq_Len, d_model)
            classification_output = model.project(encoder_output) # (B, 1, label_size)
            classification_output = classification_output.transpose(0, 1)
            classification_output = torch.squeeze(classification_output, dim=0)

            output = threshold(classification_output, 0.5)
            per_epoch_outputs.append(output)
            per_epoch_labels.append(label)
            
            print(output, label)
            label = label.float()
            classification_output = classification_output.float()
            print(classification_output)
            loss = loss_fn(classification_output, label)
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            wandb.log({"batch": batch, "loss": loss})

            loss.backward()

            optimizer.step()
            scheduler.step()
            

            global_step += 1

        #print(confusion_matrix(per_epoch_labels, per_epoch_outputs))
        val_accuracy = validate(model, val_dataloader, device)        
        


        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            model_filename = get_weights_path(config, f'{epoch:02d}')
            torch.save({
                'epoch' : epoch,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'global_step' : global_step
            }, model_filename)
        
