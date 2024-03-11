import torch
import torch.nn as nn
from diagonal_model import LongformerSelfAttention

class LongformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout_rate, layer_id):
        super(LongformerLayer, self).__init__()

        self.self_attention = LongformerSelfAttention(hidden_size, num_heads, dropout_rate, layer_id)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout_rate),
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None, global_attention_mask=None):
        attention_output = self.self_attention(x, mask, global_attention_mask)
        x = self.norm1(x + self.dropout1(attention_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x
    
class ClassificationHead(nn.Module):
    
    def __init__(self, d_model, op_label_size):
        super().__init__()
        self.proj = nn.Linear(d_model, op_label_size)
    
    def forward(self,x):
        x = torch.mean(x, dim=1)
        linfin = nn.Linear(768, 1)
        x = linfin(x)
        print(x.shape)
        act_fun = nn.Sigmoid()
        return  act_fun(x)