import torch
import torch.nn as nn
import torch.nn.functional as F


class Classical(nn.Module):
    def __init__(self, num_classes=2):
        super(Classical, self).__init__()
        self.conv = nn.Conv2d(1, 4, 2, stride=2)
        self.fc1 = nn.Linear(4*14*14, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x, y):
        preds = self.predict(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(preds, y)
        return loss

    def predict(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = self.conv(x)
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x