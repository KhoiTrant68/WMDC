import argparse

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from models.WMDC import WMDC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WMDC(N=192, M=320, num_slices=5).to(device)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device)["state_dict"]
    )
    model.eval()
    model.update(force=True)

    img = Image.open(args.image).convert("RGB")
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out_enc = model.compress(x)
        # We hook into `decompress` where GaussianConditional processes the bitstrings
        latents = []

        def hook_fn(module, input, output):
            # output is the strictly quantized `y_hat_slice` reconstructed from bits
            y_hat_slice = output.detach().cpu()
            spatial_heatmap = torch.mean(torch.abs(y_hat_slice), dim=1).squeeze(0)
            latents.append(spatial_heatmap)

        hook_handle = model.gaussian_conditional.register_forward_hook(hook_fn)
        _ = model.decompress(
            out_enc["strings"], out_enc["shape"], out_enc["original_shape"]
        )
        hook_handle.remove()

    fig, axes = plt.subplots(1, 6, figsize=(18, 3))
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    for i in range(5):
        hm = latents[i].numpy()
        axes[i + 1].imshow(hm, cmap="magma", interpolation="nearest")
        axes[i + 1].set_title(f"Slice {i+1} Latent")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.savefig("latent_sparsity_visualization.pdf", bbox_inches="tight")
    print(
        "Saved authentic sparsity visualization using the quantized bitstream variables."
    )


if __name__ == "__main__":
    main()
