import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os
from model import OneHotCVAE


def load_vae(ckpt_path, device):
    ckpt = torch.load(os.path.join(ckpt_path, "ckpts/ckpt.pt"), map_location=device)
    config = ckpt["config"]
    vae = OneHotCVAE(
        x_dim=config.x_dim,
        h_dim1=config.h_dim1,
        h_dim2=config.h_dim2,
        z_dim=config.z_dim,
    ).to(device)
    vae.load_state_dict(ckpt["model"])
    vae.eval()
    return vae, config


@torch.no_grad()
def extract_latents(vae, dataloader, device):
    zs, labels = [], []
    for data, label in dataloader:
        data, label = data.to(device), label.to(device)
        label_oh = F.one_hot(label, 10).float()
        mu, log_var = vae.encoder(data.view(-1, 784), label_oh)
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        zs.append(z.cpu().numpy())
        labels.append(label.cpu().numpy())
    return np.concatenate(zs, axis=0), np.concatenate(labels, axis=0)


def plot_latent_embeddings(zs, labels, save_path, title="t-SNE latent space"):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    z_tsne = tsne.fit_transform(zs)

    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(
        z_tsne[:, 0],
        z_tsne[:, 1],
        c=labels,
        cmap="viridis",
        s=5,
        alpha=0.7
    )
    plt.colorbar(scatter)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved t-SNE plot to {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Change these paths ===
    base_ckpt_path = "./results/mnist/2025_10_26_225755"
    forget_ckpt_path = "./results/mnist/2025_10_27_022203"

    # Load models
    vae_base, config_base = load_vae(base_ckpt_path, device)
    vae_forget, config_forget = load_vae(forget_ckpt_path, device)

    # Load test data
    test_dataset = datasets.MNIST(
        "./dataset", train=False, transform=transforms.ToTensor(), download=True
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False)

    # Extract latent representations
    zs_base, labels = extract_latents(vae_base, test_loader, device)
    zs_forget, _ = extract_latents(vae_forget, test_loader, device)

    # Plot t-SNE for both
    plot_latent_embeddings(zs_base, labels, save_path=os.path.join(base_ckpt_path, "latent_tsne_before.png"), title="Before Forgetting")
    plot_latent_embeddings(zs_forget, labels, save_path=os.path.join(forget_ckpt_path, "latent_tsne_after.png"), title="After Forgetting")


if __name__ == "__main__":
    main()
