import torch
from torch import nn


class SimpleLNPTransformer(nn.Module):
    def __init__(
        self,
        component_embedding_dim,
        num_component_types,
        num_classes,
        embed_dim=256,
        layers=2,
        heads=4,
        dropout=0.1,
    ):
        super().__init__()
        self.component_proj = nn.Linear(component_embedding_dim, embed_dim)
        self.percent_proj = nn.Linear(1, embed_dim)
        self.type_embed = nn.Embedding(num_component_types, embed_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.normal_(self.cls, std=0.02)

    def forward(self, component_embeddings, percents, component_types, mask):
        x = (
            self.component_proj(component_embeddings)
            + self.percent_proj(percents.unsqueeze(-1))
            + self.type_embed(component_types)
        )

        batch = x.shape[0]
        cls = self.cls.expand(batch, -1, -1)
        x = torch.cat([cls, x], dim=1)

        cls_mask = torch.ones(batch, 1, dtype=torch.bool, device=mask.device)
        keep_mask = torch.cat([cls_mask, mask], dim=1)
        padding_mask = ~keep_mask

        x = self.transformer(x, src_key_padding_mask=padding_mask)
        cls_rep = self.norm(x[:, 0])
        return self.head(cls_rep), cls_rep


def soft_cross_entropy(logits, target):
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(target * log_probs).sum(dim=-1).mean()
