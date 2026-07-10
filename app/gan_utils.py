import base64
import io
from pathlib import Path

import torch
from torchvision import transforms

from app.gan_model import Generator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NOISE_DIM = 100
MODEL_PATH = Path("models/generator.pth")

generator = Generator(noise_dim=NOISE_DIM).to(device)


def load_generator() -> None:
    """
    Load the trained generator model.

    The model file is created by running:
        python train_gan.py
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "models/generator.pth was not found. "
            "Please train the GAN first by running: python train_gan.py"
        )

    generator.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    generator.eval()


def generate_digit_image_base64() -> str:
    """
    Generate one MNIST-style digit and return it as a base64 encoded PNG image.
    """
    load_generator()

    with torch.no_grad():
        noise = torch.randn(1, NOISE_DIM).to(device)
        generated_image = generator(noise)

        # Convert from [-1, 1] to [0, 1].
        generated_image = (generated_image + 1) / 2
        generated_image = generated_image.clamp(0, 1)

        # Remove batch dimension: (1, 1, 28, 28) -> (1, 28, 28)
        image_tensor = generated_image.squeeze(0).cpu()

        to_pil = transforms.ToPILImage()
        image = to_pil(image_tensor)

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        return base64.b64encode(buffer.read()).decode("utf-8")
