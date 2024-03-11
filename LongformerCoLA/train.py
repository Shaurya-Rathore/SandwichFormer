import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from datasets import load_dataset
from pathlib import Path
from dataset import CoLADataset
from model import build_longformer
from config import get_weights_path, get_config
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR
import wandb
##get wandb here
wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 5,
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

def get_ds(config):
    train_data_raw = load_dataset("glue", "cola", split='train')
    val_data_raw = load_dataset("glue", "cola", split='validation')
    tokenizer = get_or_build_tokenizer(config)

    train_ds = CoLADataset(train_data_raw, tokenizer, config['seq_len'])
    val_ds = CoLADataset(val_data_raw, tokenizer, config['seq_len'])
    
    max_len = 0
    
    for item in train_data_raw:
        src_id = tokenizer.encode(item['sentence']).ids
        max_len = max(max_len, len(src_id))
    
    print(f"Max Length for sentence = {max_len}")
        
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle = True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle = True)
    
    return train_dataloader, val_dataloader, tokenizer

def get_model(config, inp_vocab_size):
    model = build_longformer(
        inp_vocab_size, 
        config['num_labels'], 
        config['seq_len'], 
        config['d_model'], 
        config['num_layers'], 
        config['num_heads'], 
        config['dropout'], 
        config['intermediate_size']
        )
    return model

def threshold(tensor, threshold_value):
    # Convert values above threshold to 1 and below or equal to threshold to 0
    return torch.where(tensor > threshold_value, torch.tensor(1), torch.tensor(0))


def train_model(config):
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "cpu"
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    train_dataloader, val_dataloader, tokenizer = get_ds(config)
    model = get_model(config, tokenizer.get_vocab_size()).to(device)
    
    
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
    
    for epoch in range(init_epoch, config['epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            optimizer.zero_grad(set_to_none=True)

            
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            label = batch['label'].to(device) # (B, 1)
            
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, Seq_Len, d_model)
            classification_output = model.project(encoder_output) # (B, 1, label_size)
            classification_output = classification_output.transpose(0, 1)
            classification_output = torch.squeeze(classification_output, dim=0)

            output = threshold(classification_output, 0.5)

            
            print(output, label)
            label = label.float()
            classification_output = classification_output.float()
            loss = loss_fn(classification_output, label)
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            wandb.log({"batch": batch, "loss": loss})

            loss.backward()

            optimizer.step()
            scheduler.step()
            

            global_step += 1
        
        model_filename = get_weights_path(config, f'{epoch:02d}')
        torch.save({
            'epoch' : epoch,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'global_step' : global_step
        }, model_filename)
        
if __name__ == '__main__':
    config = get_config()
    train_model(config)
    