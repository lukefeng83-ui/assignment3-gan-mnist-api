import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Generator architecture required by Assignment 3.

    Input:
        Noise vector with shape (batch_size, 100)

    Architecture:
        Linear: 100 -> 7 * 7 * 128
        Reshape to (batch_size, 128, 7, 7)
        ConvTranspose2d: 128 -> 64, kernel_size=4, stride=2, padding=1
        BatchNorm2d
        ReLU
        ConvTranspose2d: 64 -> 1, kernel_size=4, stride=2, padding=1
        Tanh

    Output:
        Image with shape (batch_size, 1, 28, 28)
    """

    def __init__(self, noise_dim: int = 100):
        super().__init__()

        self.noise_dim = noise_dim

        self.fc = nn.Linear(noise_dim, 7 * 7 * 128)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(-1, 128, 7, 7)
        x = self.net(x)
        return x


class Discriminator(nn.Module):
    """
    Discriminator architecture required by Assignment 3.

    Input:
        Image with shape (batch_size, 1, 28, 28)

    Architecture:
        Conv2d: 1 -> 64, kernel_size=4, stride=2, padding=1
        LeakyReLU(0.2)
        Conv2d: 64 -> 128, kernel_size=4, stride=2, padding=1
        BatchNorm2d
        LeakyReLU(0.2)
        Flatten
        Linear: 128 * 7 * 7 -> 1
        Sigmoid

    Output:
        Real/fake probability with shape (batch_size, 1)
    """

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.LeakyReLU(0.2),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
