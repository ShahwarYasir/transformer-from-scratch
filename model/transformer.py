# Full Transformer model (encoder-decoder), Section 3
import torch.nn as nn

from model.decoder import Decoder
from model.embeddings import Embeddings, PositionalEncoding
from model.encoder import Encoder


class Transformer(nn.Module):
    """
    Full encoder-decoder Transformer (paper, Section 3, Figure 1).

    Flow:
      src ids -> src embed + PE -> Encoder -> memory
      tgt ids -> tgt embed + PE -> Decoder(memory) -> decoder output
      decoder output -> Linear(d_model -> vocab) -> logits

    Softmax is intentionally left out of forward(); training uses
    nn.CrossEntropyLoss / label smoothing on the raw logits (see chunk 10),
    and generation applies softmax explicitly at decode time (chunk 14).
    """

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        h=8,
        d_ff=2048,
        n_layers=6,
        dropout=0.1,
        max_len=5000,
    ):
        super().__init__()
        self.src_embed = Embeddings(src_vocab_size, d_model)
        self.tgt_embed = Embeddings(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        self.encoder = Encoder(d_model, h, d_ff, dropout, n_layers)
        self.decoder = Decoder(d_model, h, d_ff, dropout, n_layers)

        self.generator = nn.Linear(d_model, tgt_vocab_size)

        # Weight sharing between tgt embedding and the output projection (paper, Section 3.4)
        self.generator.weight = self.tgt_embed.lut.weight

        self._init_params()

    def _init_params(self):
        # Xavier init for all params with dim > 1, as commonly used for this architecture
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask=None):
        x = self.pos_enc(self.src_embed(src))
        return self.encoder(x, src_mask)

    def decode(self, tgt, memory, src_mask=None, tgt_mask=None):
        x = self.pos_enc(self.tgt_embed(tgt))
        return self.decoder(x, memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory = self.encode(src, src_mask)
        out = self.decode(tgt, memory, src_mask, tgt_mask)
        return self.generator(out)  # (B, n_tgt, tgt_vocab_size) — raw logits