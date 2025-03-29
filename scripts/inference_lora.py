import torch
from diffusers import StableDiffusionPipeline

def generate_image(model_path, prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Base Model
    model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)

    # Load Fine-tuned LoRA Weights
    model.text_encoder.load_state_dict(torch.load(model_path, map_location=device))

    # Generate Image
    image = model(prompt).images[0]
    image.show()
    image.save("generated_image.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained LoRA model")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for image generation")
    args = parser.parse_args()

    generate_image(args.model, args.prompt)
