import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
def extract_latents_twice(vae, dataloader, device):
    zs, labels = [], []
    for data, label in dataloader:
        data, label = data.to(device), label.to(device)
        label_oh = F.one_hot(label, 10).float()

        # 1. Encode input
        mu, _ = vae.encoder(data.view(-1, 784), label_oh)

        # 2. Decode mu
        x_hat = vae.decoder(mu, label_oh)

        # 3. Encode reconstructed x_hat
        z_new, _ = vae.encoder(x_hat, label_oh)

        zs.append(x_hat.cpu().numpy())
        labels.append(label.cpu().numpy())

    return np.concatenate(zs, axis=0), np.concatenate(labels, axis=0)



def plot_latent_embeddings(zs, labels, save_path, title="t-SNE latent space"):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    z_tsne = tsne.fit_transform(zs)

    # Convert labels to numpy array (in case they're a list or tensor)
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    # Pick a colormap with distinct colors
    cmap = cm.get_cmap('tab20', n_classes)  # 'tab10' or 'tab20' are great for categories
    colors = cmap(range(n_classes))

    plt.figure(figsize=(8, 8))
    for i, label in enumerate(unique_labels):
        idx = labels == label
        plt.scatter(
            z_tsne[idx, 0],
            z_tsne[idx, 1],
            color=colors[i],
            label=str(label),
            s=10,
            alpha=0.8
        )

    plt.title(title)
    plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved t-SNE plot to {save_path}")



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Change these paths ===
    forget_ckpt_path = "./results/mnist/2025_10_28_183119"

    # Load models
    vae_forget, config_forget = load_vae(forget_ckpt_path, device)

    # Load test data
    test_dataset = datasets.MNIST(
        "./dataset", train=False, transform=transforms.ToTensor(), download=True
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False)

    # Extract latent representations
    zs_forget, forget_labels = extract_latents_twice(vae_forget, test_loader, device)

    # Plot t-SNE for both
    plot_latent_embeddings(zs_forget, forget_labels, save_path=os.path.join(forget_ckpt_path, "latent_tsne_after.png"), title="After Forgetting")


if __name__ == "__main__":
    main()
