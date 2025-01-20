import argparse
import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 32 * 32 * 3),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z).view(z.size(0), 3, 32, 32)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 3, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Training Function
def train(generator, discriminator, dataloader, optimizer_G, optimizer_D, loss_fn, epochs, latent_dim, device, output_dir):
    for epoch in range(epochs):
        for i, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # Train Discriminator
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_images = generator(z)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            optimizer_D.zero_grad()
            real_loss = loss_fn(discriminator(real_images), real_labels)
            fake_loss = loss_fn(discriminator(fake_images.detach()), fake_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            g_loss = loss_fn(discriminator(fake_images), real_labels)
            g_loss.backward()
            optimizer_G.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}] Batch {i}/{len(dataloader)} "+
                      f"Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}")

        # Save generated images
        os.makedirs(output_dir, exist_ok=True)
        save_image(fake_images.data[:25], os.path.join(output_dir, f"epoch_{epoch + 1}.png"), nrow=5, normalize=True)

# Main Script
def main():
    parser = argparse.ArgumentParser(description="Train a GAN on image datasets")
    parser.add_argument('--dataset', type=str, required=True, choices=['mnist', 'svhn', 'cifar10', 'cifar100'], help="Dataset to use")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the dataset")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    parser.add_argument('--learning_rate', type=float, default=2e-4, help="Learning rate")
    parser.add_argument('--latent_dim', type=int, default=100, help="Dimensionality of the latent space")
    parser.add_argument('--output_dir', type=str, default="output", help="Directory to save generated images")

    args = parser.parse_args()

    # Dataset setup
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
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
    generator = Generator(latent_dim=args.latent_dim).to(device)
    discriminator = Discriminator().to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    loss_fn = nn.BCELoss()

    # Train
    os.makedirs(args.output_dir, exist_ok=True)
    train(generator, discriminator, dataloader, optimizer_G, optimizer_D, loss_fn, args.epochs, args.latent_dim, device, args.output_dir)

    # Save models
    torch.save(generator.state_dict(), os.path.join(args.output_dir, 'generator.pth'))
    torch.save(discriminator.state_dict(), os.path.join(args.output_dir, 'discriminator.pth'))
    print(f"Models saved to {args.output_dir}")

if __name__ == "__main__":
    main()
