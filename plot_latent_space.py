import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
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
def latent_traversal(vae, config, save_path, n_steps=10):
    """Generate 2D latent traversal grid like the left plot."""
    device = next(vae.parameters()).device
    z_dim = config.z_dim

    # Only works cleanly if z_dim >= 2
    z1 = np.linspace(-3, 3, n_steps)
    z2 = np.linspace(-3, 3, n_steps)
    grid_imgs = []

    label = torch.tensor([0]).to(device)  # choose any label (or loop over)
    c = F.one_hot(label, 10).float().repeat(n_steps * n_steps, 1)

    for i in z1:
        for j in z2:
            z = torch.zeros((1, z_dim)).to(device)
            z[0, 0] = i
            z[0, 1] = j
            with torch.no_grad():
                out = vae.decoder(z, c[0:1]).view(-1, 1, 28, 28)
                grid_imgs.append(out.cpu())

    grid_imgs = torch.cat(grid_imgs, dim=0)
    grid = make_grid(grid_imgs, nrow=n_steps)
    save_image(grid, save_path)
    print(f"Saved latent traversal grid to {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Change these paths ===
    ckpt_path = "./results/mnist/2025_10_27_022203"  # or your SA model path

    vae, config = load_vae(ckpt_path, device)

    # === Left Plot: Latent Traversal ===
    latent_traversal(vae, config, save_path=os.path.join(ckpt_path, "latent_traversal0.png"))

    # === Right Plot: Latent Space (t-SNE) ===
    # test_dataset = datasets.MNIST("./dataset", train=False, transform=transforms.ToTensor(), download=True)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False)
    # zs, labels = extract_latents(vae, test_loader, device)
    # plot_latent_embeddings(zs, labels, save_path=os.path.join(ckpt_path, "latent_tsne.png"))


if __name__ == "__main__":
    main()
