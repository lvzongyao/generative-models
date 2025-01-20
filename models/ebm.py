import argparse
import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# Energy-Based Model (EBM)
class EBM(nn.Module):
    def __init__(self):
        super(EBM, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1)
        )

    def forward(self, x):
        return self.model(x)

# Langevin Dynamics Sampler
def langevin_dynamics(model, z, steps, step_size, noise_scale):
    z = z.clone().detach().requires_grad_(True)
    for _ in range(steps):
        energy = model(z)
        grad = torch.autograd.grad(energy.sum(), z, create_graph=True)[0]
        z = z - step_size * grad + noise_scale * torch.randn_like(z)
    return z.detach()

# Training Function
def train(ebm, dataloader, optimizer, epochs, device, latent_dim, sample_steps, step_size, noise_scale, output_dir):
    for epoch in range(epochs):
        for i, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # Initialize noise samples
            z = torch.randn(batch_size, 3, 32, 32).to(device)
            z = langevin_dynamics(ebm, z, sample_steps, step_size, noise_scale)

            # Energy loss
            optimizer.zero_grad()
            energy_real = ebm(real_images).mean()
            energy_fake = ebm(z).mean()
            loss = energy_real - energy_fake
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}] Batch {i}/{len(dataloader)} Loss: {loss.item():.4f}")

        # Save generated images
        os.makedirs(output_dir, exist_ok=True)
        save_image(z.data[:25], os.path.join(output_dir, f"epoch_{epoch + 1}.png"), nrow=5, normalize=True)

# Main Script
def main():
    parser = argparse.ArgumentParser(description="Train an Energy-Based Model (EBM) on image datasets")
    parser.add_argument('--dataset', type=str, required=True, choices=['mnist', 'svhn', 'cifar10', 'cifar100'], help="Dataset to use")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the dataset")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--latent_dim', type=int, default=100, help="Dimensionality of the latent space")
    parser.add_argument('--sample_steps', type=int, default=10, help="Number of Langevin sampling steps")
    parser.add_argument('--step_size', type=float, default=0.01, help="Step size for Langevin dynamics")
    parser.add_argument('--noise_scale', type=float, default=0.005, help="Noise scale for Langevin dynamics")
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
    ebm = EBM().to(device)
    optimizer = optim.Adam(ebm.parameters(), lr=args.learning_rate)

    # Train
    os.makedirs(args.output_dir, exist_ok=True)
    train(ebm, dataloader, optimizer, args.epochs, device, args.latent_dim, args.sample_steps, args.step_size, args.noise_scale, args.output_dir)

    # Save model
    torch.save(ebm.state_dict(), os.path.join(args.output_dir, 'ebm.pth'))
    print(f"Model saved to {os.path.join(args.output_dir, 'ebm.pth')}")

if __name__ == "__main__":
    main()
