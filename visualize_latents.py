import argparse
import torch
import matplotlib.pyplot as plt
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
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["state_dict"])
    model.eval()

    img = Image.open(args.image).convert("RGB")
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)

    # Create a Hook to intercept the Gaussian Conditional outputs (the latents!)
    latents =[]
    def hook_fn(module, input, output):
        # output is (y_hat_slice, y_slice_likelihood)
        y_hat_slice = output[0].detach().cpu()
        # Average across channels to get a 2D spatial heatmap
        spatial_heatmap = torch.mean(torch.abs(y_hat_slice), dim=1).squeeze(0)
        latents.append(spatial_heatmap)

    # Register hook on the Gaussian Conditional
    hook_handle = model.gaussian_conditional.register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model(x) # Triggers the hook 5 times (once per slice)
    
    hook_handle.remove()

    # Plot Original Image and Latent Slices
    fig, axes = plt.subplots(1, 6, figsize=(18, 3))
    
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    for i in range(5):
        # Normalize heatmap for visualization
        hm = latents[i].numpy()
        axes[i+1].imshow(hm, cmap='magma', interpolation='nearest')
        axes[i+1].set_title(f"Slice {i+1} Latent")
        axes[i+1].axis("off")

    plt.tight_layout()
    plt.savefig("latent_sparsity_visualization.pdf", bbox_inches='tight')
    print("Saved latent_sparsity_visualization.pdf. Notice how Slices 4 and 5 are mostly empty!")

if __name__ == "__main__":
    main()