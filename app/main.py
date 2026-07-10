from fastapi import FastAPI, HTTPException

from app.gan_utils import generate_digit_image_base64


app = FastAPI(
    title="Assignment 3 GAN MNIST API",
    description="FastAPI deployment of a PyTorch GAN trained on MNIST.",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "Assignment 3 GAN MNIST API is running.",
        "generate_digit_endpoint": "/generate-digit"
    }


@app.get("/generate-digit")
def generate_digit():
    """
    Generate a handwritten digit image using the trained GAN generator.

    Returns:
        A base64 encoded PNG image.
    """
    try:
        image_base64 = generate_digit_image_base64()
        return {
            "message": "Generated handwritten digit image",
            "image_format": "png",
            "image_base64": image_base64
        }
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
