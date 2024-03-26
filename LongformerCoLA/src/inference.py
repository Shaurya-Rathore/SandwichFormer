from utils.dataset import CoLADataset
from datasets import load_dataset
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from utils.train import get_ds, validate
from utils.config import get_weights_path
from utils.model import build_transformer
from trainmodel import test
import argparse
from tokenizers import Tokenizer


parser = argparse.ArgumentParser()
parser.add_argument('--ds_dir', type=str, required=False, default='./Files/spam.csv')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--d_model', type=int, default=768)
parser.add_argument('--num_heads', type=int, default=12)
parser.add_argument('--intermediate_size', type=int, default=3072)
parser.add_argument('--num_embeddings', type=int, default=30522)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--num_labels', type=int, default=2)
parser.add_argument('--lr', type=float, default=2e-9)
parser.add_argument('--epochs', type=int, default=12)
parser.add_argument('--device', type=str, required=False, default='cuda')
parser.add_argument('--seq_len', type= int, required=False, default= 512)
arguments = parser.parse_args()

config = test(arguments)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_, _, _, test_dataloader = get_ds(config, device)

tokenizer_path = Path(config['tokenizer_file'])
tokenizer = Tokenizer.from_file(str(tokenizer_path))

model = build_transformer(
        tokenizer.get_vocab_size(), 
        config['num_labels'], 
        config['seq_len'], 
        config['d_model'], 
        config['num_layers'], 
        config['num_heads'], 
        config['dropout'], 
        config['intermediate_size'],
        device
        )
model.to(device)


model_filename = get_weights_path(config, '00')
state = torch.load(model_filename)
model.load_state_dict(state['model_state_dict'])

print(validate(model, test_dataloader, device))