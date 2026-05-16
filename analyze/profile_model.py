import torch
from calflops import calculate_flops

from models.WMDC import WMDC


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Profile WMDC model")
    parser.add_argument("--N", type=int, default=192)
    parser.add_argument("--M", type=int, default=320)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument(
        "--content-adaptive",
        action="store_true",
        default=False,
        dest="use_content_adaptive",
        help="Use content-adaptive K-means token permutation in Mamba blocks.",
    )
    parser.add_argument(
        "--cluster-num",
        type=int,
        default=8,
        dest="cluster_num",
        help="Number of clusters for content-adaptive K-means tokenization.",
    )
    parser.add_argument(
        "--use-wls-shortcut",
        action="store_true",
        default=False,
        dest="use_wls_shortcut",
        help="Enable WLS/iWLS multi-scale shortcuts (must match checkpoint).",
    )
    args = parser.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(f"Device selected: {device}")

    print("Initializing WMDC model...")
    model = WMDC(
        N=args.N,
        M=args.M,
        num_slices=5,
        use_content_adaptive=args.use_content_adaptive,
        cluster_num=args.cluster_num,
        use_wls_shortcut=args.use_wls_shortcut,
    ).to(device)
    model.eval()

    batch_size = 1
    input_shape = (batch_size, 3, args.resolution, args.resolution)

    print("\nRunning a dummy forward pass to check model execution...")
    dummy_input = torch.randn(input_shape).to(device)
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print("✅ Forward pass successful!")
        print(f"   Input shape:  {dummy_input.shape}")
        print(f"   Output shape: {output['x_hat'].shape}")
    except Exception as e:
        print(f"❌ Forward pass failed with error: {e}")
        return

    print("\nCalculating FLOPs, MACs, and Params (this may take a few seconds)...")
    flops, macs, params = calculate_flops(
        model=model, input_shape=input_shape, output_as_string=True, output_precision=4
    )

    # 5. Print Results
    print("\n" + "=" * 40)
    print("          MODEL PROFILING RESULTS         ")
    print("=" * 40)
    print(f"Model:  WMDC(N={args.N}, M={args.M})")
    print(f"Image:  {args.resolution}x{args.resolution} RGB")
    print("-" * 40)
    print(f"FLOPs:  {flops}")
    print(f"MACs:   {macs}")
    print(f"Params: {params}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    main()
