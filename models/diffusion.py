import argparse
import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# Define the Diffusion Model
class DiffusionModel(nn.Module):
    def __init__(self):
        super(DiffusionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1)
        )

    def forward(self, x, t):
        return self.model(x)

# Noise Scheduler
def linear_beta_schedule(timesteps):
    return torch.linspace(1e-4, 0.02, timesteps)

# Forward Diffusion Process
def forward_diffusion_sample(x_0, t, betas):
    noise = torch.randn_like(x_0)
    alpha_t = (1 - betas).cumprod(dim=0)[t].view(-1, 1, 1, 1).to(x_0.device)
    return torch.sqrt(alpha_t) * x_0 + torch.sqrt(1 - alpha_t) * noise, noise

# Loss Function
def diffusion_loss(model, x_0, t, betas):
    x_t, noise = forward_diffusion_sample(x_0, t, betas)
    predicted_noise = model(x_t, t)
    return nn.functional.mse_loss(predicted_noise, noise)

# Training Function
def train(model, dataloader, optimizer, epochs, device, timesteps, output_dir):
    betas = linear_beta_schedule(timesteps).to(device)
    for epoch in range(epochs):
        for i, (images, _) in enumerate(dataloader):
            images = images.to(device)
            t = torch.randint(0, timesteps, (images.size(0),), device=device).long()

            optimizer.zero_grad()
            loss = diffusion_loss(model, images, t, betas)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}] Batch {i}/{len(dataloader)} Loss: {loss.item():.4f}")

        # Save example images
        os.makedirs(output_dir, exist_ok=True)
        sampled_images, _ = forward_diffusion_sample(images, timesteps - 1, betas)
        save_image(sampled_images.data[:25], os.path.join(output_dir, f"epoch_{epoch + 1}.png"), nrow=5, normalize=True)

# Main Script
def main():
    parser = argparse.ArgumentParser(description="Train a Diffusion Model on image datasets")
    parser.add_argument('--dataset', type=str, required=True, choices=['mnist', 'svhn', 'cifar10', 'cifar100'], help="Dataset to use")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the dataset")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--timesteps', type=int, default=1000, help="Number of diffusion timesteps")
    parser.add_argument('--output_dir', type=str, default="output", help="Directory to save generated images")

    args = parser.parse_args()

    # Dataset setup
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    if args.dataset == 'mnist':
        dataset = datasets.MNIST(args.data_path, train=True, download=True, transform=transform)
    elif args.dataset == 'svhn':
        dataset = datasets.SVHN(args.data_path, split='train', download=True, transform=transform)
    elif args.dataset == 'cifar10':
        dataset = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transform)
    elif args.dataset == 'cifar100':
        dataset = datasets.CIFAR100(args.data_path, train=True, download=True, transform=transform)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DiffusionModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train
    os.makedirs(args.output_dir, exist_ok=True)
    train(model, dataloader, optimizer, args.epochs, device, args.timesteps, args.output_dir)

    # Save model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'diffusion_model.pth'))
    print(f"Model saved to {os.path.join(args.output_dir, 'diffusion_model.pth')}")

if __name__ == "__main__":
    main()
