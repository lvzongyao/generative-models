# Generative Models

This repository contains implementations of various generative models including Diffusion Models, Energy-Based Models (EBM), Generative Adversarial Networks (GAN), and Variational Autoencoders (VAE).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/lvzongyao/generative-models.git
   cd generative-models
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

<!--
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
-->

## Usage

Each model can be trained using the corresponding script in the `models` directory. Below are the instructions for each model:

### Variational Autoencoder (VAE)
```bash
python models/vae.py --dataset <dataset> --data_path <path_to_data> --epochs <num_epochs> --batch_size <batch_size> --learning_rate <learning_rate> --latent_dim <latent_dim> --output_dir <output_dir>
```
#### For example:
```bash
python models/vae.py --dataset cifar10 --data_path ./data --epochs 20 --batch_size 64 --learning_rate 0.001 --latent_dim 20 --output_dir ./output
```

### Generative Adversarial Network (GAN)
```bash
python models/gan.py --dataset <dataset> --data_path <path_to_data> --epochs <num_epochs> --batch_size <batch_size> --learning_rate <learning_rate> --latent_dim <latent_dim> --output_dir <output_dir>
```
#### For example:
```bash
python models/gan.py --dataset cifar10 --data_path ./data --epochs 20 --batch_size 64 --learning_rate 0.0002 --latent_dim 100 --output_dir ./output
```

### Energy-Based Model (EBM)
```bash
python models/ebm.py --dataset <dataset> --data_path <path_to_data> --epochs <num_epochs> --batch_size <batch_size> --learning_rate <learning_rate> --latent_dim <latent_dim> --sample_steps <sample_steps> --step_size <step_size> --noise_scale <noise_scale> --output_dir <output_dir>
```
#### For example:
```bash
python models/ebm.py --dataset cifar10 --data_path ./data --epochs 20 --batch_size 64 --learning_rate 0.0001 --sample_steps 10 --step_size 0.01 --noise_scale 0.005 --output_dir ./output
```

### Diffusion Model
```bash
python models/diffusion.py --dataset <dataset> --data_path <path_to_data> --epochs <num_epochs> --batch_size <batch_size> --learning_rate <learning_rate> --timesteps <timesteps> --output_dir <output_dir>
```
#### For example:
```bash
python models/diffusion.py --dataset cifar10 --data_path ./data --epochs 20 --batch_size 64 --learning_rate 0.0001 --timesteps 1000 --output_dir ./output
```

## Model Descriptions

### Variational Autoencoder (VAE) [[paper]](https://arxiv.org/abs/1312.6114)
A VAE is a type of generative model that learns to encode data into a latent space and then decode it back to the original space. The model is trained to maximize the evidence lower bound (ELBO) on the data likelihood.

### Generative Adversarial Network (GAN) [[paper]](https://arxiv.org/abs/1406.2661)
A GAN consists of two neural networks, a generator and a discriminator, that are trained together. The generator learns to generate realistic data, while the discriminator learns to distinguish between real and generated data.

### Energy-Based Model (EBM) [[paper]](https://www.researchgate.net/profile/Marcaurelio-Ranzato/publication/216792742_A_Tutorial_on_Energy-Based_Learning/links/0912f50c6862425435000000/A-Tutorial-on-Energy-Based-Learning.pdf)
An energy-based model is a type of generative model that learns to assign low energy to data points from the data distribution and high energy to other points. The model is trained using a contrastive divergence algorithm.

### Diffusion Model [[paper]](https://arxiv.org/abs/2006.11239)
A diffusion model is a type of generative model that learns to generate data by reversing a diffusion process. The model is trained to denoise data that has been progressively corrupted by noise.

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
