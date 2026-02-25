import argparse
import torch
import math
import torch.nn.functional as F
from PIL import Image

# Force matplotlib to not use any Xwindows backend (prevents showing/crashing)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms

from models.WMDC import WMDC

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Visual Comparison Patches for WMDC")
    parser.add_argument("-i", "--image", type=str, required=True, help="Path to input image (e.g., kodim04.png)")
    parser.add_argument("-c", "--checkpoint", type=str, required=True, help="Path to trained model .pth.tar")
    parser.add_argument("-o", "--output", type=str, default="visual_comparison", help="Output base filename (without extension)")
    parser.add_argument("--crop", type=int, nargs=4, default=[300, 450, 250, 400], help="Crop coordinates: y1 y2 x1 x2")
    parser.add_argument("--N", type=int, default=192, help="Model channels N")
    parser.add_argument("--M", type=int, default=320, help="Model channels M")
    parser.add_argument("--cuda", action="store_true", help="Use GPU if available")
    return parser.parse_args()

def main():
    args = parse_args()
    
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Initialize and Load Model
    model = WMDC(N=args.N, M=args.M).to(device)
    print(f"Loading checkpoint: {args.checkpoint}")
    
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
        
    model.eval()
    model.update()

    # 2. Load Image
    print(f"Processing image: {args.image}")
    img = Image.open(args.image).convert('RGB')
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)
    
    # 3. Pad Image (must be divisible by 128 for Mamba/Attention blocks)
    pad_h = (128 - x.size(2) % 128) % 128
    pad_w = (128 - x.size(3) % 128) % 128
    x_padded = F.pad(x, (0, pad_w, 0, pad_h))
    
    # 4. Forward Pass & Metrics
    with torch.no_grad():
        out_net = model(x_padded)
        # Crop back to original dimensions and clamp to [0, 1] valid range
        x_hat = out_net["x_hat"][:, :, :x.size(2), :x.size(3)].clamp(0, 1)
        
        mse = F.mse_loss(x, x_hat)
        psnr = -10 * math.log10(mse.item()) if mse.item() > 0 else 100
        
        num_pixels = x.size(2) * x.size(3)
        bpp = sum(torch.log(lh).sum() for lh in out_net["likelihoods"].values()) / (-math.log(2) * num_pixels)

    # 5. Setup plot variables
    orig_np = x.squeeze().permute(1, 2, 0).cpu().numpy()
    rec_np = x_hat.squeeze().permute(1, 2, 0).cpu().numpy()

    y1, y2, x1, x2 = args.crop
    
    # Safety check if crop is out of bounds
    if y2 > orig_np.shape[0] or x2 > orig_np.shape[1] or y1 >= y2 or x1 >= x2:
        print(f"Warning: Invalid crop {args.crop} for image size {orig_np.shape}. Plotting full image.")
        y1, y2, x1, x2 = 0, orig_np.shape[0], 0, orig_np.shape[1]

    # 6. Plot the Patches
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(orig_np[y1:y2, x1:x2])
    axes[0].set_title("Original Image (Crop)", fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(rec_np[y1:y2, x1:x2])
    axes[1].set_title(f"WMDC (Ours)\nBPP: {bpp.item():.3f} | PSNR: {psnr:.2f}dB", fontsize=14)
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # 7. Save to PDF and JPG (bbox_inches='tight' removes excess white margins)
    pdf_path = f"{args.output}.pdf"
    jpg_path = f"{args.output}.jpg"
    
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(jpg_path, format='jpg', dpi=300, bbox_inches='tight')
    
    # Close plot to free up memory (important if looping over many images later)
    plt.close(fig)
    
    print(f"Success! Saved visual comparisons to:\n -> {pdf_path}\n -> {jpg_path}")

if __name__ == "__main__":
    main()