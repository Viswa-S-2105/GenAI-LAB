import os
import replicate
import gradio as gr

def generate_image(prompt):
    try:
        output = replicate.run(
            "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
            input={
                "width": 768,
                "height": 768,
                "prompt": prompt,
                "refine": "expert_ensemble_refiner",
                "apply_watermark": False,
                "num_inference_steps": 25
            }
        )

        # Download image and return path
        for index, image_url in enumerate(output):
            image_path = f"output_{index}.png"
            image_data = replicate.utils.download(image_url)
            with open(image_path, "wb") as file:
                file.write(image_data.read())
            return image_path

    except Exception as e:
        return f"❌ Error: {str(e)}"

# Gradio UI
gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(label="Enter a prompt", placeholder="e.g., A panda surfing a tsunami in space"),
    outputs=gr.Image(label="Generated Image"),
    title="🧠 AI Image Generator (Replicate SDXL)",
    theme="soft",
).launch()
