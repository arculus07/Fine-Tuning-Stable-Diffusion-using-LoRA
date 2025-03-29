import torch
import torch.nn as nn
from torchvision import transforms, datasets
from diffusers import StableDiffusionPipeline
from transformers import AutoTokenizer, AutoModel

# Load dataset
def load_data(data_path, batch_size=2):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(data_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# Define LoRA Model
class LoRAModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.lora_adapter = nn.Linear(768, 768)

    def forward(self, x):
        x = self.base_model(x)
        x = self.lora_adapter(x)
        return x

# Training Function
def train_lora(data_path, epochs=10, batch_size=2, lr=1e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Base Model
    base_model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)

    # Apply LoRA Adapter
    model = LoRAModel(base_model.text_encoder).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    dataloader = load_data(data_path, batch_size)

    for epoch in range(epochs):
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, images)

            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "models/lora_model.pth")
    print("Training Complete. Model saved!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    train_lora(args.dataset, args.epochs, args.batch_size, args.lr)
