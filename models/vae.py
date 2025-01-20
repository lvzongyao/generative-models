import argparse
import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# VAE Model
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 3, 400),
            nn.ReLU(),
            nn.Linear(400, latent_dim * 2)  # Mean and log-variance
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 32 * 32 * 3),
            nn.Sigmoid(),
        )

        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder(x)
        mu, logvar = x[:, :self.latent_dim], x[:, self.latent_dim:]
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar

# Loss Function
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 32 * 32 * 3), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Training Function
def train(model, dataloader, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, _ in dataloader:
            x = x.to(device).view(-1, 32 * 32 * 3)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = loss_function(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader.dataset):.4f}")

# Main Script
def main():
    parser = argparse.ArgumentParser(description="Train a VAE on image datasets")
    parser.add_argument('--dataset', type=str, required=True, choices=['mnist', 'svhn', 'cifar10', 'cifar100'], help="Dataset to use")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the dataset")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--latent_dim', type=int, default=20, help="Dimensionality of the latent space")
    parser.add_argument('--output_dir', type=str, default="output", help="Directory to save generated images")

    args = parser.parse_args()

    # Dataset setup
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
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
    model = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train
    os.makedirs(args.output_dir, exist_ok=True)
    train(model, dataloader, optimizer, args.epochs, device)

    # Save model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'vae.pth'))
    print(f"Model saved to {os.path.join(args.output_dir, 'vae.pth')}")

if __name__ == "__main__":
    main()
