"""
feature reduction/preprocessing for data
Resize, PCA, Autoencoder
to dim 32, 30, 16
"""

import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF



def feature_reduc(train_x, test_x, f_type='resize', to_size=32):
    if f_type == 'resize':
        train_x = TF.resize(torch.unsqueeze(train_x, 1), (14, 14))
        test_x = TF.resize(torch.unsqueeze(test_x, 1), (14, 14))
        train_x = torch.squeeze(train_x)
        test_x = torch.squeeze(test_x)

    elif f_type == 'pca':
        train_x = train_x.reshape(train_x.shape[0], -1)
        test_x = test_x.reshape(test_x.shape[0], -1)
        pca = PCA(n_components=to_size)
        train_x = torch.tensor(pca.fit_transform(train_x))
        test_x = torch.tensor(pca.transform(test_x))

    elif f_type == 'encoder':
        class AutoEncoder(nn.Module):
            def __init__(self):
                super(AutoEncoder, self).__init__()
                self.encoder = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(784, to_size),
                    nn.ReLU()
                )
                self.decoder = nn.Sequential(
                    nn.Linear(to_size, 784),
                    nn.Sigmoid(),
                    nn.Unflatten(1, (28, 28))
                )

            def forward(self, x):
                e = self.encoder(x)
                d = self.decoder(e)
                return d

        encoder = AutoEncoder()
        criterion = nn.MSELoss()
        opt = optim.Adam(encoder.parameters())
        print('```AutoEncoder training```')
        for epoch in range(50):
            encoder.train()
            opt.zero_grad()
            outputs = encoder(train_x)
            loss = criterion(outputs, train_x)
            loss.backward()
            opt.step()
            print(f'Epoch {epoch + 1}, loss: {loss.item():.4f}')
        encoder.eval()
        with torch.no_grad():
            o = encoder(test_x)
            test_loss = criterion(o, test_x)
            print(f'Test loss: {test_loss.item():.4f}')
            print('```end```')
            train_x, test_x = encoder(train_x), encoder(test_x)

    return train_x, test_x

