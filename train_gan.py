import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from app.gan_model import Generator, Discriminator


def train_gan():
    """
    Train the Assignment 3 GAN architecture on MNIST.

    The model uses:
    - Generator: Linear + ConvTranspose2d layers
    - Discriminator: Conv2d layers

    The trained generator can then be used by the FastAPI endpoint.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    noise_dim = 100
    batch_size = 128
    learning_rate = 0.0002
    epochs = 30

    os.makedirs("models", exist_ok=True)
    os.makedirs("generated_images", exist_ok=True)

    # Tanh outputs images in [-1, 1], so real MNIST images are normalized to [-1, 1].
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    generator = Generator(noise_dim=noise_dim).to(device)
    discriminator = Discriminator().to(device)

    criterion = nn.BCELoss()

    optimizer_g = optim.Adam(
        generator.parameters(),
        lr=learning_rate,
        betas=(0.5, 0.999)
    )

    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=learning_rate,
        betas=(0.5, 0.999)
    )

    fixed_noise = torch.randn(16, noise_dim).to(device)

    for epoch in range(epochs):
        for real_images, _ in dataloader:
            real_images = real_images.to(device)
            current_batch_size = real_images.size(0)

            real_labels = torch.ones(current_batch_size, 1).to(device)
            fake_labels = torch.zeros(current_batch_size, 1).to(device)

            # ----------------------------------------------------
            # 1. Train Discriminator
            # ----------------------------------------------------
            noise = torch.randn(current_batch_size, noise_dim).to(device)
            fake_images = generator(noise)

            real_predictions = discriminator(real_images)
            fake_predictions = discriminator(fake_images.detach())

            d_loss_real = criterion(real_predictions, real_labels)
            d_loss_fake = criterion(fake_predictions, fake_labels)
            d_loss = d_loss_real + d_loss_fake

            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()

            # ----------------------------------------------------
            # 2. Train Generator
            # ----------------------------------------------------
            noise = torch.randn(current_batch_size, noise_dim).to(device)
            fake_images = generator(noise)
            fake_predictions = discriminator(fake_images)

            # Generator wants fake images to be classified as real.
            g_loss = criterion(fake_predictions, real_labels)

            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"D Loss: {d_loss.item():.4f} "
            f"G Loss: {g_loss.item():.4f}"
        )

        # Save example generated images every 10 epochs.
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                generated_samples = generator(fixed_noise)
                save_image(
                    generated_samples,
                    f"generated_images/epoch_{epoch + 1}.png",
                    nrow=4,
                    normalize=True
                )

    torch.save(generator.state_dict(), "models/generator.pth")
    torch.save(discriminator.state_dict(), "models/discriminator.pth")

    print("Training complete.")
    print("Saved generator to models/generator.pth")
    print("Saved discriminator to models/discriminator.pth")


if __name__ == "__main__":
    train_gan()
