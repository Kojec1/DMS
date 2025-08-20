import argparse
import os
import random
import math
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

from nn.modules.model import MHModel
from nn.metrics.gaze_metrics import angular_error as angular_error_metric
from data.dataset import MPIIFaceGazeMatDataset
from data.augmentation import apply_clahe


def build_model(backbone: str, in_channels: int, num_bins: int, device: torch.device) -> MHModel:
	model = MHModel(
		num_landmarks=6,
		pretrained_backbone=False,
		in_channels=in_channels,
		dropout_rate=0.0,
		num_bins=num_bins,
		backbone=backbone,
	).to(device)
	model.eval()
	return model


def load_weights(model: MHModel, ckpt_path: str, device: torch.device) -> None:
	checkpoint = torch.load(ckpt_path, map_location=device)
	state_dict = checkpoint.get('model_state_dict', checkpoint)
	model.load_state_dict(state_dict, strict=True)


def get_expected_angles_from_logits(yaw_logits: torch.Tensor, pitch_logits: torch.Tensor, bin_width: float) -> Tuple[torch.Tensor, torch.Tensor]:
	b, num_bins = yaw_logits.shape
	offsets = torch.arange(num_bins, device=yaw_logits.device).float() - (num_bins - 1) / 2.0
	yaw_probs = F.softmax(yaw_logits, dim=1)
	pitch_probs = F.softmax(pitch_logits, dim=1)
	yaw_deg = bin_width * torch.sum(yaw_probs * offsets, dim=1)
	pitch_deg = bin_width * torch.sum(pitch_probs * offsets, dim=1)
	return pitch_deg, yaw_deg


def build_transforms(img_size: int, in_channels: int) -> transforms.Compose:
	if in_channels == 1:
		normalize = transforms.Normalize(mean=[0.485], std=[0.229])
		return transforms.Compose([
			transforms.Resize((img_size, img_size)),
			transforms.ToTensor(),
			normalize,
		])
	else:
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		return transforms.Compose([
			transforms.Resize((img_size, img_size)),
			transforms.ToTensor(),
			normalize,
		])


def load_sample_rgb_from_mat(dataset: MPIIFaceGazeMatDataset, file_path: str, local_idx: int, target_size: int) -> Tuple[Image.Image, np.ndarray, np.ndarray]:
	img_pil, gaze_2d_angles_np, _head_pose_angles_np, landmarks_np = dataset._load_from_mat(file_path, local_idx)
	eff_w, eff_h = img_pil.size
	if isinstance(target_size, int):
		target_w = target_h = target_size
	else:
		target_h, target_w = target_size
	if (eff_w, eff_h) != (target_w, target_h):
		scale_x = target_w / eff_w
		scale_y = target_h / eff_h
		img_pil = img_pil.resize((target_w, target_h), Image.BILINEAR)
		landmarks_np = landmarks_np.copy()
		landmarks_np[:, 0] *= scale_x
		landmarks_np[:, 1] *= scale_y
	return img_pil, gaze_2d_angles_np, landmarks_np


def make_model_input_from_pil(img_pil: Image.Image, in_channels: int, use_clahe: bool, img_size: int, device: torch.device) -> torch.Tensor:
	proc = img_pil
	if in_channels == 1 and proc.mode != 'L':
		proc = proc.convert('L')
	elif in_channels == 3 and proc.mode != 'RGB':
		proc = proc.convert('RGB')
	if use_clahe:
		proc = apply_clahe(proc)
	transform = build_transforms(img_size=img_size, in_channels=in_channels)
	tensor = transform(proc).unsqueeze(0).to(device)
	return tensor


def compute_anchor_from_landmarks(landmarks_np: np.ndarray) -> Tuple[float, float]:
	left_eye_center = landmarks_np[[0, 1], :].mean(axis=0)
	right_eye_center = landmarks_np[[2, 3], :].mean(axis=0)
	anchor = (left_eye_center + right_eye_center) / 2.0
	return float(anchor[0]), float(anchor[1])


def draw_gaze_arrows(ax, img: Image.Image, anchor_xy: Tuple[float, float], gt_pitch_yaw_rad: Tuple[float, float], pred_pitch_yaw_deg: Tuple[float, float], length: float = 40.0) -> None:
	ax.imshow(img)
	ax.axis('off')
	x0, y0 = anchor_xy
	gt_pitch_deg = math.degrees(float(gt_pitch_yaw_rad[0]))
	gt_yaw_deg = math.degrees(float(gt_pitch_yaw_rad[1]))
	gt_pitch_rad = math.radians(gt_pitch_deg)
	gt_yaw_rad = math.radians(gt_yaw_deg)
	pred_pitch_deg, pred_yaw_deg = float(pred_pitch_yaw_deg[0]), float(pred_pitch_yaw_deg[1])
	pred_pitch_rad = math.radians(pred_pitch_deg)
	pred_yaw_rad = math.radians(pred_yaw_deg)
	def vec_from_angles(pitch_rad: float, yaw_rad: float) -> Tuple[float, float]:
		dx = -math.sin(yaw_rad) * math.cos(pitch_rad) * length
		dy = -math.sin(pitch_rad) * length
		return dx, dy
	dx_gt, dy_gt = vec_from_angles(gt_pitch_rad, gt_yaw_rad)
	dx_pr, dy_pr = vec_from_angles(pred_pitch_rad, pred_yaw_rad)
	ax.arrow(x0, y0, dx_gt, dy_gt, color='red', width=0.0, head_width=8, head_length=10, length_includes_head=True, alpha=0.8)
	ax.arrow(x0, y0, dx_pr, dy_pr, color='lime', width=0.0, head_width=8, head_length=10, length_includes_head=True, alpha=0.8)


def find_sample_for_subject(
		mat_dataset: MPIIFaceGazeMatDataset,
		file_path: str,
		model: MHModel,
		in_channels: int,
		use_clahe: bool,
		img_size: int,
		mode: str,
		threshold: float,
		bin_width: float,
		scan_limit: Optional[int],
		device: torch.device,
		rng: random.Random,
	) -> Tuple[Image.Image, Tuple[float, float], Tuple[float, float], float, np.ndarray]:
	import h5py
	with h5py.File(file_path, 'r') as f:
		n_samples = f['Data']['label'].shape[0]
	indices = list(range(n_samples))
	if mode == 'random':
		rng.shuffle(indices)
	else:
		pass

	best_idx = None
	best_err = None
	best_record = None

	to_scan = n_samples if (scan_limit is None or scan_limit <= 0) else min(scan_limit, n_samples)

	for k in range(to_scan):
		idx = indices[k]
		rgb_img, gt_angles_rad_np, landmarks_np = load_sample_rgb_from_mat(mat_dataset, file_path, idx, target_size=img_size)
		model_input = make_model_input_from_pil(rgb_img, in_channels=in_channels, use_clahe=use_clahe, img_size=img_size, device=device)
		with torch.no_grad():
			pred_landmarks_logits, yaw_logits, pitch_logits = model(model_input)
			pred_pitch_deg_t, pred_yaw_deg_t = get_expected_angles_from_logits(yaw_logits, pitch_logits, bin_width=bin_width)
			pred_pitch_deg = float(pred_pitch_deg_t.item())
			pred_yaw_deg = float(pred_yaw_deg_t.item())
		gt_pitch_deg = math.degrees(float(gt_angles_rad_np[0]))
		gt_yaw_deg = math.degrees(float(gt_angles_rad_np[1]))
		err = float(angular_error_metric(
			torch.tensor([pred_pitch_deg], dtype=torch.float32),
			torch.tensor([pred_yaw_deg], dtype=torch.float32),
			torch.tensor([gt_pitch_deg], dtype=torch.float32),
			torch.tensor([gt_yaw_deg], dtype=torch.float32),
		).item())

		match = False
		if mode == 'below':
			match = err < threshold
		elif mode == 'above':
			match = err > threshold
		elif mode == 'random':
			match = True
		else:
			raise ValueError(f"Unknown mode: {mode}")

		if match:
			return rgb_img, (pred_pitch_deg, pred_yaw_deg), (gt_angles_rad_np[0], gt_angles_rad_np[1]), err, landmarks_np

		# Track best candidate in case we don't find a match
		if best_err is None or (
			(mode == 'below' and err < best_err) or (mode == 'above' and err > best_err)
		):
			best_err = err
			best_record = (rgb_img, (pred_pitch_deg, pred_yaw_deg), (gt_angles_rad_np[0], gt_angles_rad_np[1]), err, landmarks_np)
			best_idx = idx

	if best_record is not None:
		return best_record
	rgb_img, gt_angles_rad_np, landmarks_np = load_sample_rgb_from_mat(mat_dataset, file_path, 0, target_size=img_size)
	model_input = make_model_input_from_pil(rgb_img, in_channels=in_channels, use_clahe=use_clahe, img_size=img_size, device=device)
	with torch.no_grad():
		pred_landmarks_logits, yaw_logits, pitch_logits = model(model_input)
		pred_pitch_deg_t, pred_yaw_deg_t = get_expected_angles_from_logits(yaw_logits, pitch_logits, bin_width=bin_width)
		pred_pitch_deg = float(pred_pitch_deg_t.item())
		pred_yaw_deg = float(pred_yaw_deg_t.item())
	gt_pitch_deg = math.degrees(float(gt_angles_rad_np[0]))
	gt_yaw_deg = math.degrees(float(gt_angles_rad_np[1]))
	err = float(angular_error_metric(
		torch.tensor([pred_pitch_deg], dtype=torch.float32),
		torch.tensor([pred_yaw_deg], dtype=torch.float32),
		torch.tensor([gt_pitch_deg], dtype=torch.float32),
		torch.tensor([gt_yaw_deg], dtype=torch.float32),
	).item())
	return rgb_img, (pred_pitch_deg, pred_yaw_deg), (gt_angles_rad_np[0], gt_angles_rad_np[1]), err, landmarks_np


def locate_run_dirs(runs_root: str) -> List[Tuple[int, str]]:
	entries = []
	if not os.path.isdir(runs_root):
		raise FileNotFoundError(f"Runs root not found: {runs_root}")
	for name in os.listdir(runs_root):
		full = os.path.join(runs_root, name)
		if not os.path.isdir(full):
			continue
		if name.startswith('run_val_'):
			# Try to parse X as int, supporting pXX as well
			suffix = name[len('run_val_'):]
			try:
				if suffix.startswith('p') and len(suffix) == 3:
					subject_id = int(suffix[1:])
				else:
					subject_id = int(suffix)
				entries.append((subject_id, full))
			except ValueError:
				continue
	entries.sort(key=lambda x: x[0])
	return entries


def main(args: argparse.Namespace) -> None:
	rng = random.Random(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
	print(f"Using device: {device}")

	run_entries = locate_run_dirs(args.runs_root)
	if len(run_entries) == 0:
		raise RuntimeError(f"No run_val_X directories found under {args.runs_root}")

	model = build_model(args.backbone, args.input_channels, args.num_bins, device=device)

	mat_dataset_helper = MPIIFaceGazeMatDataset(
		dataset_path=args.data_dir,
		participant_ids=[0],
		transform=None,
		input_channels=3,
		use_cache=False,
		use_clahe=False,
		downscale_size=None,
	)

	cols = 5
	rows = math.ceil(len(run_entries) / cols)
	fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.5 * rows))
	axes = np.array(axes).reshape(rows, cols)

	for idx, (subject_id, run_dir) in enumerate(run_entries):
		row = idx // cols
		col = idx % cols
		ax = axes[row, col]

		ckpt_path = os.path.join(run_dir, 'best_model.pth')
		if not os.path.isfile(ckpt_path):
			ax.set_title(f"S{subject_id:02d}: missing")
			ax.axis('off')
			continue
		print(f"Subject {subject_id:02d}: loading {ckpt_path}")
		load_weights(model, ckpt_path, device=device)

		mat_path = os.path.join(args.data_dir, f"p{subject_id:02d}.mat")
		if not os.path.isfile(mat_path):
			ax.set_title(f"S{subject_id:02d}: no MAT")
			ax.axis('off')
			continue

		rgb_img, pred_deg, gt_rad, err_deg, landmarks_np = find_sample_for_subject(
			mat_dataset=mat_dataset_helper,
			file_path=mat_path,
			model=model,
			in_channels=args.input_channels,
			use_clahe=args.use_clahe,
			img_size=args.img_size,
			mode=args.mode,
			threshold=args.threshold,
			bin_width=args.angle_bin_width,
			scan_limit=args.scan_limit,
			device=device,
			rng=rng,
		)

		anchor = compute_anchor_from_landmarks(landmarks_np)
		draw_gaze_arrows(ax, rgb_img, anchor, gt_rad, pred_deg)
		ax.set_title(f"S{subject_id:02d}  MAE={err_deg:.1f}°", fontsize=10)

	for j in range(idx + 1, rows * cols):
		row = j // cols
		col = j % cols
		axes[row, col].axis('off')

	plt.suptitle("Leave-One-Out Visualization (GT:red, Pred:green)")
	plt.tight_layout(rect=[0, 0.03, 1, 0.95])
	if args.output:
		os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
		plt.savefig(args.output, dpi=200)
		print(f"Saved figure to {args.output}")
	else:
		plt.show()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Leave-One-Out Gaze Visualization')
	parser.add_argument('--runs_root', type=str, required=True, help='Directory containing run_val_X subdirectories with best_model.pth')
	parser.add_argument('--data_dir', type=str, required=True, help='Directory containing MPIIFaceGaze .mat files (p00.mat ... p14.mat)')
	parser.add_argument('--backbone', type=str, default='mobilenet', choices=['mobilenet', 'convnext', 'tinyvit'], help='Backbone used during training')
	parser.add_argument('--input_channels', type=int, default=1, choices=[1, 3], help='Input channels used during training (1=grayscale)')
	parser.add_argument('--img_size', type=int, default=224, help='Inference image size')
	parser.add_argument('--num_bins', type=int, default=14, help='Number of bins for yaw/pitch heads')
	parser.add_argument('--angle_bin_width', type=float, default=3.0, help='Bin width in degrees for expected angle computation')
	parser.add_argument('--use_clahe', action='store_true', help='Apply CLAHE to model input (recommended to mirror training)')
	parser.add_argument('--mode', type=str, default='random', choices=['below', 'above', 'random'], help='Selection mode for images per subject')
	parser.add_argument('--threshold', type=float, default=5.0, help='Angular error threshold (degrees) for below/above modes')
	parser.add_argument('--scan_limit', type=int, default=2000, help='Max samples to scan per subject (<=0 for all)')
	parser.add_argument('--seed', type=int, default=42, help='Random seed')
	parser.add_argument('--cpu', action='store_true', help='Force CPU even if CUDA is available')
	parser.add_argument('--output', type=str, default='leave_one_out_visualization.png', help='Path to save the composed figure (omit to show interactively)')
	args = parser.parse_args()
	main(args) 