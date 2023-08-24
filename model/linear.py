from torch import nn
import torch.nn.functional as F

class NonLinearClassifier(nn.Module):
    def __init__(self, feat_dim=512, num_classes=10):
        super(NonLinearClassifier, self).__init__()
        self.fc1 = nn.Linear(feat_dim, feat_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(feat_dim, feat_dim)
        self.fc3 = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        features = F.relu(self.fc1(features))
        features = F.relu(self.fc2(self.dropout(features)))
        return self.fc3(features)
