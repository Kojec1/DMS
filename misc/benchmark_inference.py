import argparse
import time
import statistics
import torch

from nn.modules.model import MHModel


def measure_device(backbone: str, in_channels: int, img_size: int, batch_size: int, device: torch.device, warmup_iters: int, iters: int) -> tuple[float, float]:
    model = MHModel(
        num_landmarks=6,
        pretrained_backbone=False,
        in_channels=in_channels,
        dropout_rate=0.0,
        num_bins=32,
        num_theta_bins=32,
        num_phi_bins=60,
        backbone=backbone,
    ).to(device)
    model.eval()

    # Dummy input
    x = torch.randn(batch_size, in_channels, img_size, img_size, device=device)

    # Warmup
    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        for _ in range(warmup_iters):
            _ = model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()

    # Timed runs
    times_ms: list[float] = []
    with torch.no_grad():
        for _ in range(iters):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    mean_ms = statistics.fmean(times_ms)
    std_ms = statistics.pstdev(times_ms)
    return mean_ms, std_ms


def main():
    parser = argparse.ArgumentParser(description='Benchmark MHModel inference latency on CPU and CUDA (if available).')
    parser.add_argument('--backbone', type=str, default='mobilenet', choices=['mobilenet', 'convnext', 'tinyvit'], help='Backbone to benchmark')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size (pixels)')
    parser.add_argument('--input_channels', type=int, default=1, choices=[1, 3], help='Number of input channels')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for dummy input')
    parser.add_argument('--warmup_iters', type=int, default=50, help='Number of warmup iterations')
    parser.add_argument('--iters', type=int, default=1000, help='Number of timed iterations')
    args = parser.parse_args()

    # Enable cuDNN benchmarking for consistent input shapes
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True

    print(f"Benchmarking backbone={args.backbone}, img_size={args.img_size}, C={args.input_channels}, batch_size={args.batch_size}")

    # CPU
    cpu_device = torch.device('cpu')
    cpu_mean, cpu_std = measure_device(
        backbone=args.backbone,
        in_channels=args.input_channels,
        img_size=args.img_size,
        batch_size=args.batch_size,
        device=cpu_device,
        warmup_iters=args.warmup_iters,
        iters=args.iters,
    )
    print(f"CPU: mean={cpu_mean:.3f} ms, std={cpu_std:.3f} ms over {args.iters} runs")

    # GPU (if available)
    if torch.cuda.is_available():
        gpu_device = torch.device('cuda:0')
        gpu_name = torch.cuda.get_device_name(gpu_device)
        print(f"CUDA detected: {gpu_name}")
        gpu_mean, gpu_std = measure_device(
            backbone=args.backbone,
            in_channels=args.input_channels,
            img_size=args.img_size,
            batch_size=args.batch_size,
            device=gpu_device,
            warmup_iters=args.warmup_iters,
            iters=args.iters,
        )
        print(f"CUDA: mean={gpu_mean:.3f} ms, std={gpu_std:.3f} ms over {args.iters} runs")
    else:
        print("CUDA not available; skipping GPU benchmark.")


if __name__ == '__main__':
    main() 