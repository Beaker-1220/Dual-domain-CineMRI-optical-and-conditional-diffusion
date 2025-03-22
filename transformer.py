import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=400000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        # 计算输入张量的序列长度
        seq_len = x.size(1) * x.size(2)
        
        # 将位置编码扩展到与输入张量相同的批量大小
        pe = self.pe[:, :seq_len].repeat(x.size(0), 1, 1)
        x = x.reshape(x.size(0), x.size(1)*x.size(2), x.size(3))
        
        # 添加位置编码并应用dropout
        x = x + pe
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        assert self.head_dim * nhead == d_model

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)
        q = self.q_linear(query).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_linear(key).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if attn_mask is not None:
            scores += attn_mask
        attn = F.softmax(scores, dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_linear(out)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, q, k, v, src_mask=None):
        src2 = self.self_attn(q, k, v, attn_mask=src_mask)
        src = self.norm1(q + src2)
        src2 = self.linear2(self.dropout1(F.relu(self.linear1(src))))
        src = self.norm2(src + src2)
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, src_mask=None):
        for layer in self.layers:
            src = layer(q, k, v, src_mask)
        return self.norm(src)

class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, input_dim, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(num_layers, d_model, nhead, dim_feedforward, dropout)
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.qkv_layer = MultiHeadAttention(d_model, nhead) # 添加qkv层
        self.linear_layer = nn.Linear(d_model, input_dim)

    def forward(self, image, flow1, flow2):
        
        # 对输入图像和光流特征图进行嵌入
        img_embedded = self.embedding(image)  # [batch_size, seq_length, d_model]
        flow1_embedded = self.embedding(flow1)  # [batch_size, seq_length, d_model]
        flow2_embedded = self.embedding(flow2)# [batch_size, seq_length, d_model]
        # img_embedded = img_embedded.reshape(img_embedded.size(0), img_embedded.size(1)*img_embedded.size(2), img_embedded.size(3))
        # flow1_embedded = flow1_embedded.reshape(flow1_embedded.size(0), flow1_embedded.size(1)*flow1_embedded.size(2), flow1_embedded.size(3))
        # flow2_embedded = flow2_embedded.reshape(flow2_embedded.size(0), flow2_embedded.size(1)*flow2_embedded.size(2), flow2_embedded.size(3))
        
        # 对图像和光流特征图进行位置编码
        q = img_embedded
        k = torch.cat([flow1_embedded, flow2_embedded], dim=1)
        v = img_embedded
        combined = torch.cat([q, k, v], dim=1)  # [batch_size, 3*seq_length, d_model]
        combined = self.positional_encoding(combined)
        # 假设 combined 的形状为 [batch_size, seq_length, d_model]
        batch_size, total_length, d_model = combined.shape
        q_length = total_length // 4
        v_length = total_length // 4
        k_length = total_length // 4

        # 使用 torch.split 按照计算出的长度进行分割
        q, k_0, k_1, v = torch.split(combined, [q_length, k_length, k_length, v_length], dim=1)
        transformer_output = self.encoder(q, k_0, v)
        transformer_output = self.encoder(transformer_output, k_1, transformer_output)
        output = self.linear_layer(transformer_output)

        return output
    
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, src_mask=None):
        for layer in self.layers:
            src = layer(q, k, v, src_mask)
        return self.norm(src)

class Transformer_dual(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, input_dim, dropout=0.1):
        super(Transformer_dual, self).__init__()
        self.encoder = TransformerEncoder(num_layers, d_model, nhead, dim_feedforward, dropout)
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.qkv_layer = MultiHeadAttention(d_model, nhead) # 添加qkv层
        self.linear_layer = nn.Linear(d_model, input_dim)

    def forward(self, image, dct_image):
        
        # 对输入图像和光流特征图进行嵌入
        img_embedded = self.embedding(image)  # [batch_size, seq_length, d_model]
        dct_image = self.embedding(dct_image)  # [batch_size, seq_length, d_model]


        # 对图像和光流特征图进行位置编码
        q = img_embedded
        k = dct_image
        v = img_embedded
        combined = torch.cat([q, k, v], dim=1)  # [batch_size, 3*seq_length, d_model]
        combined = self.positional_encoding(combined)
        # 假设 combined 的形状为 [batch_size, seq_length, d_model]
        batch_size, total_length, d_model = combined.shape
        q_length = total_length // 3
        v_length = total_length // 3
        k_length = total_length // 3

        # 使用 torch.split 按照计算出的长度进行分割
        q, k, v = torch.split(combined, [q_length, k_length, v_length], dim=1)
        transformer_output = self.encoder(q, k, v)
        output = self.linear_layer(transformer_output)

        return output