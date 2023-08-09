import torch.nn as nn

class Class_Mapping_Vanila(nn.Module):
    def __init__(
        self, n_classes, feature_size, use_mlp=False
    ):
        super().__init__()
        self.use_mlp = use_mlp
        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(feature_size, feature_size),
                nn.InstanceNorm1d(feature_size),
                nn.GELU(),
                nn.Linear(feature_size, feature_size),
            )
        self.class_embeddings = nn.Embedding(n_classes, feature_size)  
        
    def forward(self, src, class_vector):
        b, c, h, w, d = src.shape
        class_embedding = self.class_embeddings(class_vector)
        if self.use_mlp:
            class_embedding = self.mlp(class_embedding)
        assert b==1, 'only supports batch size 1'
        # [b,1,feat] @ [1,feat,dim], batch dimension become class_embedding batch dimension.
        masks = (class_embedding @ src.view(b, c, h * w * d)).view(-1, 1, h, w, d)
        return masks, class_embedding
