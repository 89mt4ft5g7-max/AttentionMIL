import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMIL(nn.Module):

    def __init__(self, input_dim, hidden_dim, attention_dim):

        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

        # Attention
        self.attention_V = nn.Linear(hidden_dim, attention_dim)
        self.attention_w = nn.Linear(attention_dim, 1)

        # Classifier
        self.classifier = nn.Linear(hidden_dim, 3)

    def forward(self, x, return_attention=False):

        # Encoder
        h = self.encoder(x)

        # Attention
        A = self.attention_w(
            torch.tanh(
                self.attention_V(h)
            )
        )

        a = F.softmax(A, dim=0)

        # Pooling
        z = torch.sum(a * h, dim=0)

        # Prediction
        y = self.classifier(z)

        if return_attention:
            return y, a

        return y