import math
import torch 
import torch.nn as nn
import torch.nn.functional as F

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(seq_len, d_model)
        
        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) 
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module): 
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.d_model = d_model
        self.linear_1 = nn.Linear(d_model,d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff,d_model)
    
    def forward(self, x):
        #(batch,seq_len,d_model)--->(batch,seq_len,d_ff)--->(batch,seq_len,d_model)
        return self.linear_2(self.dropout(F.gelu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, h, dropout):
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:

            attention_scores.masked_fill_(mask == 0, -1e9)
            
        attention_scores = attention_scores.softmax(dim=-1)
       
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v) 

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        
        return self.w_o(x)

class ResidualConnection(nn.Module):
    
        def __init__(self, features, dropout):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization(features)
    
        def forward(self, x, sublayer):
            return x + self.dropout(sublayer(self.norm(x)))
    
class ClassificationHead(nn.Module):
    
    def __init__(self, d_model, op_label_size, device):
        super().__init__()
        self.linfin = nn.Linear(d_model, 1).to(device)  # Move to device
        self.op_label_size = op_label_size
    
    def forward(self, x):
        x = torch.mean(x, dim=1)
        x = self.linfin(x)
        act_fun = nn.Sigmoid()
        return  act_fun(x)
    
class EncoderBlock(nn.Module):
    def __init__(self, features,  AttentionHeadType, feedforward, dropout):
        super().__init__()
        self.AttentionHeadType = AttentionHeadType

        self.feedforward = feedforward
        self.residual_connections = nn.ModuleList([ResidualConnection(features,dropout) for _ in range(2)])
        
    def forward(self, x, src_mask):
        if self.AttentionHeadType is not None:
            x = self.residual_connections[0](x, lambda x: self.AttentionHeadType(x, x, x, src_mask))
        if self.feedforward is not None:
            x = self.residual_connections[1](x, self.feedforward)
        return x
    
class Encoder(nn.Module):

    def __init__(self, features, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class Transformer(nn.Module):
    def __init__(self, encoder, src_embed, src_pos, classification_head):
        super().__init__()
        self.src_embed = src_embed
        self.src_pos = src_pos
        self.encoder = encoder
        self.classification_head = classification_head
        
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def project(self, x):
        return self.classification_head(x)
    
       
def build_transformer(src_vocab_size, label_vocab_size, src_seq_len, d_model, N, h, dropout, d_ff,device):
    src_embed = InputEmbeddings(d_model, src_vocab_size).to(device)
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout).to(device)
    
    
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention = MultiHeadAttentionBlock(d_model,h,dropout).to(device)
        encoder_block = EncoderBlock(d_model, encoder_self_attention, None, dropout).to(device)
        encoder_blocks.append(encoder_block)

    for _ in range(N):
        encoder_self_attention = MultiHeadAttentionBlock(d_model,h,dropout).to(device)
        feed_forward = FeedForwardBlock(d_model, d_ff, dropout).to(device)
        encoder_block = EncoderBlock(d_model, encoder_self_attention, feed_forward, dropout).to(device)
        encoder_blocks.append(encoder_block)

    for _ in range(N):
        feed_forward = FeedForwardBlock(d_model, d_ff, dropout).to(device)
        encoder_block = EncoderBlock(d_model, None, feed_forward, dropout).to(device)
        encoder_blocks.append(encoder_block)
        
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks)).to(device)
        
    classification_head = ClassificationHead(d_model, label_vocab_size, device).to(device)
        
    transformer = Transformer(encoder, src_embed, src_pos,  classification_head).to(device)
        
    for p in transformer.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)

    print('built')
        
    return transformer
