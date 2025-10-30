# attack_sa_vae.py
import torch
import torch.nn.functional as F
from model import OneHotCVAE, Classifier
import argparse
import os
import logging
import numpy as np
from torchvision.utils import save_image, make_grid

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_ckpt", type=str, required=True, help="Path to pretrained SA-VAE checkpoint (ckpt.pt)")
    parser.add_argument("--classifier_ckpt", type=str, default="./classifier_ckpts/model.pt", help="Path to pretrained classifier")
    parser.add_argument("--labels_to_attack", nargs="+", type=int, required=True, help="Labels to generate real-looking digits for")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_iters", type=int, default=2000, help="Number of attack iterations")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for decoder update")
    parser.add_argument("--z_dim", type=int, default=8, help="Latent dimension if not in ckpt")
    parser.add_argument("--save_dir", type=str, default="./attack_results9", help="Directory to save generated images")
    parser.add_argument("--log_freq", type=int, default=100, help="Logging frequency")
    parser.add_argument("--classifier_returns_logits", action="store_true", help="If classifier returns logits")
    args = parser.parse_args()
    ckpt = torch.load(os.path.join(args.vae_ckpt, "ckpts/ckpt.pt"), map_location=device)
    return args, ckpt

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(asctime)s - %(message)s")

def generate_label_batch(batch_size, labels_to_attack, device):
    labels = torch.from_numpy(
        np.random.choice(labels_to_attack, size=batch_size)
    ).long().to(device)
    c = F.one_hot(labels, num_classes=10).float()
    return labels, c

def main():
    import numpy as np
    setup_logging()
    args,ckpt = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------- Load SA-VAE ----------
    config = ckpt.get("config", None)
    z_dim = config.z_dim

    vae = OneHotCVAE(x_dim=config.x_dim, h_dim1=config.h_dim1, h_dim2=config.h_dim2, z_dim=z_dim).to(device)
    vae.load_state_dict(ckpt["model"])
    vae.train()  # we will update decoder parameters

    # ---------- Load pretrained classifier ----------
    classifier = Classifier().to(device)
    clf_ckpt = torch.load(args.classifier_ckpt, map_location=device)
    classifier.load_state_dict(clf_ckpt)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad = False

    # Only update decoder parameters
    decoder_params = list(vae.parameters())
    optimizer = torch.optim.Adam(decoder_params, lr=args.lr)

    logging.info("Starting SA-VAE attack training...")

    for step in range(args.n_iters):
        optimizer.zero_grad()
        # generate random latent z and target label batch
        z = torch.randn((args.batch_size, z_dim), device=device)
        labels, c_target = generate_label_batch(args.batch_size, args.labels_to_attack, device)

        # forward decoder
        out = vae.decoder(z, c_target).view(-1, 1, 28, 28)

        # classifier loss
        clf_out = classifier(out)
        if args.classifier_returns_logits:
            loss = F.cross_entropy(clf_out, labels)
        else:
            loss = F.nll_loss(clf_out, labels)

        # backprop to decoder
        loss.backward()
        optimizer.step()

        if (step+1) % args.log_freq == 0:
            logging.info(f"Step {step+1}/{args.n_iters} | Classifier loss: {loss.item():.6f}")

            # save sample grid
            with torch.no_grad():
                sample_z = torch.randn((len(args.labels_to_attack)*10, z_dim), device=device)
                sample_labels = torch.arange(len(args.labels_to_attack)).repeat_interleave(10).to(device)
                sample_c = F.one_hot(sample_labels, num_classes=10).float()
                imgs = vae.decoder(sample_z, sample_c).view(-1,1,28,28)
                grid = make_grid(imgs, nrow=10)
                save_image(grid, os.path.join(args.save_dir, f"step_{step+1}.png"))

    # Save updated decoder
    torch.save({
        "decoder_state_dict": vae.state_dict(),
        "z_dim": z_dim
    }, os.path.join(args.save_dir, "vae_decoder_attacked.pt"))
    logging.info(f"Attack finished. Updated decoder saved at {os.path.join(args.save_dir, 'vae_decoder_attacked.pt')}")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()
