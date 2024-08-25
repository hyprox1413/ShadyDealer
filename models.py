import torch
from torch import nn

class PredTransformer(nn.Module):
    def __init__(self, n_dim, n_heads=8):
        super().__init__()
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(n_dim, n_heads, batch_first=True), 12)
    
    def forward(self, x):
        batch_size = x.shape[0]
        n_dim = x.shape[2]
        tgt = torch.cat([torch.ones(batch_size, 1, n_dim).to(x), x], dim=1)
        tgt = self.decoder(tgt, x, tgt_mask=nn.Transformer.generate_square_subsequent_mask(tgt.shape[1]), tgt_is_causal=True)
        next_prices = tgt[:, -1, :]
        return next_prices