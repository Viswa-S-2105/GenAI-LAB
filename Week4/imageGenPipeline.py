# Import libraries
import torch
from diffusers import DiffusionPipeline
import gradio as gr

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load Stable Diffusion XL Model (Optimized for Colab)
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16  # Lower memory usage
)
pipe.to(device)

# Function to generate images
def generate_image(prompt):
    image = pipe(prompt).images[0]  # Generate image from text
    return image

# Gradio UI
interface = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(label="Enter Text Prompt"),
    outputs=gr.Image(label="Generated Image"),
    title="Stable Diffusion XL - Text to Image",
    description="Enter a text description, and the model will generate an image.",
)

# Launch the Gradio app
interface.launch(share=True)
