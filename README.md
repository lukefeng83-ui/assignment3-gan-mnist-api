# Assignment 3: GAN MNIST API

This project implements a Generative Adversarial Network using PyTorch. The GAN follows the required Assignment 3 architecture and is trained on the MNIST dataset to generate handwritten digit images. The trained generator is then deployed through a FastAPI endpoint.

## Assignment Requirements Covered

This submission covers the three parts of Assignment 3:

1. Implement a GAN using PyTorch that matches the required architecture.
2. Train the GAN on MNIST and add it to the Module 6 API.
3. Answer the conceptual and calculation questions about GAN building blocks.

## Project Structure

```text
.
├── app
│   ├── __init__.py
│   ├── gan_model.py
│   ├── gan_utils.py
│   └── main.py
├── train_gan.py
├── part3_answers.md
├── requirements.txt
├── .gitignore
└── README.md
```

## GAN Architecture

### Generator

Input:

```text
Noise vector with shape (batch_size, 100)
```

Architecture:

```text
Linear: 100 -> 7 * 7 * 128
Reshape to: (batch_size, 128, 7, 7)

ConvTranspose2d: 128 -> 64
kernel_size=4, stride=2, padding=1
BatchNorm2d
ReLU

ConvTranspose2d: 64 -> 1
kernel_size=4, stride=2, padding=1
Tanh
```

Output:

```text
(batch_size, 1, 28, 28)
```

### Discriminator

Input:

```text
Image with shape (batch_size, 1, 28, 28)
```

Architecture:

```text
Conv2d: 1 -> 64
kernel_size=4, stride=2, padding=1
LeakyReLU(0.2)

Conv2d: 64 -> 128
kernel_size=4, stride=2, padding=1
BatchNorm2d
LeakyReLU(0.2)

Flatten
Linear: 128 * 7 * 7 -> 1
Sigmoid
```

Output:

```text
Real/fake probability
```

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

If using Python 3 explicitly:

```bash
python3 -m pip install -r requirements.txt
```

## Train the GAN

Run:

```bash
python train_gan.py
```

Or:

```bash
python3 train_gan.py
```

This will:

1. Download the MNIST dataset.
2. Train the Generator and Discriminator.
3. Save generated image examples to `generated_images/`.
4. Save the trained models to `models/`.

Expected generated files:

```text
models/generator.pth
models/discriminator.pth
generated_images/epoch_10.png
generated_images/epoch_20.png
generated_images/epoch_30.png
```

## Run the API

After training the model, run:

```bash
uvicorn app.main:app --reload
```

Or:

```bash
python3 -m uvicorn app.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## API Endpoint

### `GET /generate-digit`

This endpoint generates a handwritten digit image using the trained GAN generator.

Example response:

```json
{
  "message": "Generated handwritten digit image",
  "image_format": "png",
  "image_base64": "..."
}
```

The image is returned as a base64 encoded PNG string.

## GitHub Submission

To upload to GitHub:

```bash
git init
git add .
git commit -m "Implement Assignment 3 GAN MNIST API"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
git push -u origin main
```

Submit the GitHub repository link.
