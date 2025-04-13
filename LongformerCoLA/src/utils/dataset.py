import torch
import re
import torch.nn as nn
from torch.utils.data import Dataset
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string


class CoLADataset(Dataset):
    
    def __init__(self, ds, tokenizer, seq_len):
        super().__init__()
        self.ds = ds
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        self.sos_token = torch.tensor([tokenizer.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer.token_to_id("[PAD]")], dtype=torch.int64)
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        #lemmatizer = WordNetLemmatizer()

        pair = self.ds[index]
        src_text = pair["sentence"]
        src_label = pair['label']

        text = src_text.lower()
    
    
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
        # Tokenize the text into words
        tokens = word_tokenize(text)
    
        # Lemmatize the words
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
        # Join the lemmatized tokens back into a single string
        src_text = ' '.join(lemmatized_tokens)
        
        input_tokens = self.tokenizer.encode(src_text).ids
        
        num_padding_token = self.seq_len - len(input_tokens) - 2
        
        if num_padding_token<0:
            raise ValueError('Input sentence exceeds Sequence Length limit')
        
        enc_inp = torch.cat(
            [
                self.sos_token,
                torch.tensor(input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * num_padding_token, dtype=torch.int64),
            ],
            dim=0,
        )
        
        label = torch.tensor(src_label, dtype=torch.int64)
        
        assert enc_inp.size(0) == self.seq_len
        
        return{
            "encoder_input": enc_inp,
            "encoder_mask": (enc_inp != self.pad_token).unsqueeze(0).unsqueeze(0).int(), 
            "label": label,
        }
        
        
        

        
