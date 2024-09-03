import torch
from torch.nn import functional as F
from point_tnt.model import PointTNT

def set_requires_grad_for_layer(layer: torch.nn.Module, train: bool) -> None:
    """Sets the attribute requires_grad to True or False for each parameter.

        Args:
            layer: the layer to freeze.
            train: if true train the layer.
    """
    for p in layer.parameters():
        p.requires_grad = train

class Embedder(torch.nn.Module):
    """

    Neural network to compute image embedding.

        size_embedding: the size for the embedding.
        normalize_embedding: if True normalize the embedding using the L2 norm.
        pretrained_feature_extactor: if True use a pretrained feature extractor.
        train_feature_extactor: if True train the feature extractor.

    """
    def __init__(self,
                 size_embedding: int,
                 normalize_embedding: bool,
                 train_feature_extactor: bool) -> None:
        super().__init__()

        self.feature_ext = PointTNT(emb_dims=512)
        self.embedding = torch.nn.Linear(self.feature_ext.mlp_head[4].out_features, size_embedding)

        self.normalize_embedding = normalize_embedding
        set_requires_grad_for_layer(self.feature_ext, train_feature_extactor)

    def forward(self, anchor, positive=None, negative=None):
        def single_pass(x):
            x = self.feature_ext(x)
            embedding = self.embedding(x)

            if self.normalize_embedding:
                embedding = F.normalize(embedding, p=2, dim=1)

            return embedding

        if positive is not None and negative is not None:
            embedding_anchor = single_pass(anchor)
            embedding_positive = single_pass(positive)
            embedding_negative = single_pass(negative)

            return embedding_anchor, embedding_positive, embedding_negative
        else:
            return single_pass(anchor)