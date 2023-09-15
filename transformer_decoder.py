import torch
import torch.nn as nn
import torch.nn.functional as F


#this is an adaptation of the original SASRec's decoder
#We try to keep the same structure as the original SASRec's decoder
#In general, there is no necessity to use this version, and we can just use standard pytorch's transformer decoder
#But we want to keep the same structure as the original SASRec and gSASRec papers 
#SASRec uses somewhat weird version of multihead attention, where all the heads share the linear layers

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout_rate=0.5):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.val_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_rate) # Change the dropout rate as needed

    def forward(self, queries, keys, causality=False):
        Q = self.query_proj(queries)
        K = self.key_proj(keys)
        V = self.val_proj(keys)

        # Split and concat
        Q_ = torch.cat(Q.chunk(self.num_heads, dim=2), dim=0)
        K_ = torch.cat(K.chunk(self.num_heads, dim=2), dim=0)
        V_ = torch.cat(V.chunk(self.num_heads, dim=2), dim=0)

        # Multiplication
        outputs = torch.matmul(Q_, K_.transpose(1, 2))

        # Scale
        outputs = outputs / (K_.size(-1) ** 0.5)

        # Key Masking
        key_masks = torch.sign(torch.sum(torch.abs(keys), dim=-1))
        key_masks = key_masks.repeat(self.num_heads, 1)
        key_masks = key_masks.unsqueeze(1).repeat(1, queries.size(1), 1)
        
        outputs = outputs.masked_fill(key_masks == 0, float('-inf'))

        # Causality
        if causality:
            diag_vals = torch.ones_like(outputs[0])
            tril = torch.tril(diag_vals)
            masks = tril[None, :, :].repeat(outputs.size(0), 1, 1)

            outputs = outputs.masked_fill(masks == 0, float('-inf'))

        # Activation
        outputs = F.softmax(outputs, dim=-1)
        outputs = torch.nan_to_num(outputs, nan=0.0, posinf=0.0, neginf=0.0)


        # Query Masking
        query_masks = torch.sign(torch.sum(torch.abs(queries), dim=-1))
        query_masks = query_masks.repeat(self.num_heads, 1)
        query_masks = query_masks.unsqueeze(-1).repeat(1, 1, keys.size(1))

        outputs *= query_masks

        attention_chunks = outputs.chunk(self.num_heads, dim=0)
        attention_weights = torch.stack(attention_chunks, dim=1)


        # Dropouts
        outputs = self.dropout(outputs)

        # Weighted sum
        outputs = torch.matmul(outputs, V_)

        # Restore shape
        outputs = torch.cat(outputs.chunk(self.num_heads, dim=0), dim=2)
        return outputs, attention_weights


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim, dropout_rate=0.5, causality=True):
        super(TransformerBlock, self).__init__()
        
        self.first_norm = nn.LayerNorm(dim)
        self.second_norm = nn.LayerNorm(dim)
        
        self.multihead_attention = MultiHeadAttention(dim, num_heads, dropout_rate)
        
        self.dense1 = nn.Linear(dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, dim)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.causality = causality
        
    def forward(self, seq, mask=None):
        x = self.first_norm(seq)
        queries = x
        keys = seq
        x, attentions = self.multihead_attention(queries, keys, self.causality)
        
        # Add & Norm
        x = x + queries
        x = self.second_norm(x)
        
        # Feed Forward
        residual = x
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)
        
        # Add & Norm
        x = x + residual
        
        # Apply mask if provided
        if mask is not None:
            x *= mask
            
        return x, attentions
