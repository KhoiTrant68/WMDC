# WMDC

**Wavelet-Mamba Dictionary Compression (WMDC)** is a state-of-the-art learned image compression model that combines wavelet transforms, Visual State Space (VSS) modules, and dictionary-based coding with entropic optimal transport. The model achieves superior rate-distortion performance through three key innovations:

1. **Frequency-Disentangled Mamba (FDM)**: Efficiently captures long-range dependencies while preserving high-frequency details
2. **Spatially-Adaptive EOT Dictionary Attention**: Dynamically adapts dictionary utilization using unbalanced optimal transport
3. **Markovian Slice-Based Context Model**: Autoregressive entropy modeling with latent residual prediction

## Installation
    
### Prerequisites
- Python 3.11
- CUDA 11.8+ (for GPU support)
- torch, torchvision

### Setup

1. Clone the repository and navigate to the project directory:
```bash
cd WMDC
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install vmamba (selective scan operations):
```bash
cd vmamba
pip install -e .
cd ..
```

## Usage

### Training

Basic training command:
```bash
python train.py -d /path/to/dataset --save_path checkpoints
```

Single GPU training:
```bash
python train.py \
    -d /path/to/dataset \
    --save_path checkpoints \
    --epochs 100 \
    --batch-size 8 \
    --lr 1e-4 \
    --aux-lr 1e-3 \
    --lambda 0.0130 \
    --patch-size 256 \
    --metric mse \
    --seed 2026
```

Multi-GPU distributed training:
```bash
acceleate launch train.py \
    -d /path/to/dataset \
    --save_path checkpoints \
    --epochs 10 \
    --batch-size 8 \
    --lambda 0.0018 \
    --routing_mode softmax
```


**Training Arguments:**
- `-d, --dataset`: Path to image dataset (required)
- `--save_path`: Directory to save checkpoints (default: `checkpoints`)
- `-e, --epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 8)
- `--lr`: Learning rate (default: 1e-4)
- `--aux-lr`: Auxiliary learning rate (default: 1e-3)
- `--lambda`: Rate-distortion weight (default: None)
- `--patch-size`: Training patch size (default: 256)
- `--clip_max_norm`: Gradient clipping (default: 1.0)
- `--checkpoint`: Path to resume from checkpoint
- `--metric`: Loss metric - `mse` or `ms-ssim` (default: `mse`)
- `--routing_mode`: Routing mode - `softmax`, `balanced_eot`, or `unbalanced_eot` (default: `softmax`)
- `--seed`: Random seed (default: 2026)

### Evaluation

Evaluate on a dataset:
```bash
python eval.py \
    --dataset /path/to/test/images \
    --checkpoint /path/to/checkpoint.pth \
    --output results \
    --cuda
```

Example with Kodak dataset:
```bash
python eval.py \
    --dataset /kaggle/input/datasets/kodak-test \
    --checkpoint ./checkpoints/lambda_0.0018_mse/checkpoint_best.pth.tar \
    --output kodak \
    --cuda \
    --routing_mode softmax
```

**Evaluation Arguments:**
- `--dataset`: Path to test images (required)
- `--checkpoint`: Path to model checkpoint (required)
- `--output`: Output directory for results (default: `output`)
- `--cuda`: Use GPU for inference
- `--routing_mode`: Routing mode - `softmax`, `balanced_eot`, or `unbalanced_eot` (default: `softmax`)

### Visualization

Visualize attention maps on image directory:
```bash
python analyze/visualize_attention.py \
    --img_dir /path/to/image \
    --checkpoint /path/to/checkpoint.pth \
    --slice 3 \
    --cuda \
    -o attention_maps.pdf \
    --routing_mode softmax
```

Visualize latent features:
```bash
python analyze/visualize_latents.py \
    --image /path/to/image.png \
    --checkpoint /path/to/checkpoint.pth \
    --routing_mode softmax \
    --output latent_visualization
```

Visualize patches:
```bash
python analyze/visualize_patches.py \
    -i /path/to/image.png \
    -c /path/to/checkpoint.pth \
    --cuda \
    --routing_mode softmax \
    -o visual_comparison
```


Rate-distortion plot:
```bash
python plot_rd.py --checkpoint /path/to/checkpoint.pth --dataset /path/to/dataset
```

Profile model performance:
```bash
python analyze/profile_model.py \
    --checkpoint /path/to/checkpoint.pth \
    --input_size 256 \
    --cuda
```

Analyze bit allocation:
```bash
python analyze/analysis_bit_allocation.py \
    --checkpoint /path/to/checkpoint.pth \
    --dataset /path/to/dataset \
    --output bit_analysis_results
```

Plot bitrate-distortion latency:
```bash
python analyze/plot_bd_latency.py \
    --results_dir /path/to/results \
    --output bd_latency_plot.png
```

## Dataset Preparation

The training script expects images in standard image formats (PNG, JPEG).
For best results, the dataset should be organized as:
```
dataset/
├── image1.png
├── image2.png
├── ...
└── imageN.png
```

## Model Architecture

The WMDC model combines:
- Wavelet transforms for multi-scale decomposition
- Visual State Space (VSS) modules for feature encoding
- Multi-scale distribution coding for entropy modeling
- Spatial context models (CSM) with Triton backend optimization

See `models/WMDC.py` for implementation details.

## Output

Training outputs:
- Checkpoints saved to `--save_path` directory
- Logs saved to `logs/` directory
- TensorBoard logs for training visualization

Evaluation outputs:
- Metrics saved to results directory
- Compressed images saved for visual inspection
- Rate-distortion data

## Performance Metrics

The model outputs:
- **PSNR**: Peak Signal-to-Noise Ratio
- **MS-SSIM**: Multi-Scale Structural Similarity
- **BPP**: Bits Per Pixel (actual compressed size)
- **Loss**: Rate-distortion loss value


## References

- CompressAI: https://github.com/InterDigitalInc/CompressAI
- Vision State Space: https://github.com/MzeroMiko/VMamba.git


## License

See LICENSE file for details.
