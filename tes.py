import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self,embed_dim,dropout=0.1):
        super(SelfAttention,self).__init__()
        self.embed_dim = embed_dim

        self.query = nn.Linear(embed_dim,embed_dim)
        self.key = nn.Linear(embed_dim,embed_dim)
        self.value = nn.Linear(embed_dim,embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(embed_dim)

    def forward(self,x, mask=None):
        batch_size, seq_len, embed_dim = x.shape
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = torch.bmm(Q,K.transpose(1,2))/self.scale
        if mask is not None:
            scores = scores.masked_fill(mask==0,-1e9)
        
        attention_weight = F.softmax(scores,dim=-1)
        attention_weight = self.dropout(attention_weight)

        output = torch.bmm(attention_weight,V)
        return output,attention_weight
    
class MutilHeadAttention(nn.Module):
    def __init__(self,embed_dim,num_heads,dropout=0.1):
        super(MutilHeadAttention,self).__init__()
        assert embed_dim%num_heads ==0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim//num_heads

        self.query = nn.Linear(embed_dim,embed_dim)
        self.key = nn.Linear(embed_dim,embed_dim)
        self.value = nn.Linear(embed_dim,embed_dim)
        self.output_linear = nn.Linear(embed_dim,embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(embed_dim)

    def forward(self,x,mask=None):
        batch_size,seq_len,embed_dim = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        K = K.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        V = V.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)

        scores = torch.matmul(Q,K.transpose(-2,-1))/self.scale
        if mask is not None :
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask==0,-1e9)

        attention_weights = F.softmax(scores,dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights,V)
        context = context.transpose(1,2).contiguous().view(
            batch_size,seq_len,embed_dim
        )
        output = self.output_linear(context)

        return output,attention_weights