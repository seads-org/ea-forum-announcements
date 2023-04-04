from torch import nn


class PostClassifier(nn.Module):
    def __init__(self, embed_dim=768, num_classes=3):
        super(PostClassifier, self).__init__()
        self.head = nn.Linear(embed_dim, num_classes)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.head.weight.data.uniform_(-initrange, initrange)
        self.head.bias.data.zero_()

    def forward(self, x):
        return self.head(x)
