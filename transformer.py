import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
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
        x = x + self.pe[:, :x.size(1)]
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

    def forward(self, src, src_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)
        src = self.norm1(src + src2)
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

    def forward(self, src, src_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask)
        return self.norm(src)

class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, input_dim, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(num_layers, d_model, nhead, dim_feedforward, dropout)
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

    def forward(self, image, flow1, flow2):
        # 对输入图像和光流特征图进行嵌入
        img_embedded = self.embedding(image)  # [batch_size, seq_length, d_model]
        flow1_embedded = self.embedding(flow1)  # [batch_size, seq_length, d_model]
        flow2_embedded = self.embedding(flow2)  # [batch_size, seq_length, d_model]

        # 将嵌入结果合并
        combined = img_embedded + flow1_embedded + flow2_embedded  # 这里可以使用其他方式合并

        # 添加位置编码
        combined = self.positional_encoding(combined)

        # Transformer 编码器
        transformer_output = self.encoder(combined)

        return transformer_output