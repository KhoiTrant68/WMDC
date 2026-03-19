import torch
from calflops import calculate_flops

from models.WMDC import WMDC


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device selected: {device}")

    print("Initializing WMDC model...")
    model = WMDC(N=192, M=320, num_slices=5).to(device)
    model.eval()

    batch_size = 1
    input_shape = (batch_size, 3, 256, 256)

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
    print("Model:  WMDC(N=192, M=320)")
    print("Image:  256x256 RGB")
    print("-" * 40)
    print(f"FLOPs:  {flops}")
    print(f"MACs:   {macs}")
    print(f"Params: {params}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    main()
