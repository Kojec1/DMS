from torch.utils.data import Dataset
import os
import numpy as np
import torch
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF
import random
from .augmentation import horizontal_flip, normalize_landmarks, crop_to_content, apply_clahe, random_affine_with_landmarks, landmarks_smoothing, crop_to_landmarks
import cv2
import h5py
import bisect
from tqdm import tqdm

class BaseDataset(Dataset):
    """Base PyTorch Dataset template. Custom datasets should inherit from this class."""
    def __init__(self, transform=None):
        self.transform = transform

    def __len__(self):
        # Subclasses should implement this method to return the size of the dataset.
        raise NotImplementedError("Subclasses must implement the __len__ method.")

    def __getitem__(self, index):
        # Subclasses should implement this method to return a data sample.
        raise NotImplementedError("Subclasses must implement the __getitem__ method.")
    
    
class MPIIFaceGazeDataset(BaseDataset):
    """ 
    Landmarks:
    - 0: Left outer eye
    - 1: Left inner eye
    - 2: Right inner eye
    - 3: Right outer eye
    - 4: Left mouth
    - 5: Right mouth
    """
    def __init__(self, dataset_path, participant_ids, transform=None, is_train=False, affine_aug=True, flip_aug=True, use_cache=False, label_smoothing=0.0, input_channels=1, use_clahe=False, angle_bin_width=3.0, num_angle_bins=14):
        super().__init__(transform)
        self.dataset_path = dataset_path
        self.samples = []
        self.is_train = is_train
        self.affine_aug = affine_aug
        self.flip_aug = flip_aug
        self.use_cache = use_cache
        self.label_smoothing = label_smoothing
        self.input_channels = input_channels
        self.use_clahe = use_clahe
        self.angle_bin_width = angle_bin_width
        self.num_angle_bins = num_angle_bins
        
        if self.use_cache:
            self.image_cache = {}
            print("Image caching enabled for MPIIFaceGazeDataset.")
        
        for p_id in participant_ids:
            participant_folder = os.path.join(self.dataset_path, f"p{p_id:02d}")
            annotation_file = os.path.join(participant_folder, f"p{p_id:02d}.txt")

            if not os.path.exists(annotation_file):
                print(f"Warning: Annotation file not found {annotation_file}. Skipping participant {p_id}.")
                continue

            with open(annotation_file, 'r') as f:
                for line_idx, line in enumerate(f):
                    parts = line.strip().split(' ')
                    if len(parts) != 28:
                        print(f"Warning: Malformed line {line_idx+1} in {annotation_file}. Expected 28 parts, got {len(parts)}. Content: '{line.strip()}'. Skipping.")
                        continue

                    image_path_rel = parts[0]
                    image_path = os.path.join(participant_folder, image_path_rel)
                    
                    try:
                        # Dimensions 2-3: Gaze location on screen
                        gaze_screen_px_np = np.array([float(parts[1]), float(parts[2])], dtype=np.float32)
                        
                        # Dimensions 4-15: Facial landmarks
                        facial_landmarks_np = np.array([float(p) for p in parts[3:15]], dtype=np.float32).reshape(6, 2)
                        
                        # Dimensions 16-18: Head pose rotation vector (Rodrigues)
                        head_pose_rvec_np = np.array([float(p) for p in parts[15:18]], dtype=np.float32)
                        
                        # Dimensions 19-21: Head pose translation vector
                        head_pose_tvec_np = np.array([float(p) for p in parts[18:21]], dtype=np.float32)
                        
                        # Dimensions 22-24: Face center in camera coordinates
                        face_center_cam_np = np.array([float(p) for p in parts[21:24]], dtype=np.float32)
                        
                        # Dimensions 25-27: 3D gaze target in camera coordinates
                        gaze_target_cam_np = np.array([float(p) for p in parts[24:27]], dtype=np.float32)
                        
                        # Dimension 28: Evaluation eye
                        eval_eye_str = parts[27]

                        # Calculate 3D gaze direction vector
                        gaze_direction_cam_3d = gaze_target_cam_np - face_center_cam_np
                        
                        norm_gaze_direction_val = np.linalg.norm(gaze_direction_cam_3d)
                        
                        if norm_gaze_direction_val == 0:
                            # Handle zero vector case for gaze direction
                            normalized_gaze_vec = np.array([0.0, 0.0, -1.0], dtype=np.float32)  # Default forward direction
                            pitch = 0.0
                            yaw = 0.0
                        else:
                            normalized_gaze_vec = gaze_direction_cam_3d / norm_gaze_direction_val
                            # Pitch (vertical angle)
                            asin_arg = np.clip(-normalized_gaze_vec[1], -1.0, 1.0)
                            pitch = np.arcsin(asin_arg)
                            # Yaw (horizontal angle)
                            yaw = np.arctan2(-normalized_gaze_vec[0], -normalized_gaze_vec[2])

                        gaze_2d_angles_np = np.array([pitch, yaw], dtype=np.float32)

                        sample_data = {
                            'image_path': image_path,
                            'gaze_screen_px': gaze_screen_px_np,
                            'facial_landmarks': facial_landmarks_np,
                            'head_pose_rvec': head_pose_rvec_np,
                            'head_pose_tvec': head_pose_tvec_np,
                            'face_center_cam': face_center_cam_np,
                            'gaze_target_cam': gaze_target_cam_np,
                            'eval_eye': eval_eye_str,
                            'gaze_direction_cam_3d': normalized_gaze_vec.astype(np.float32),
                            'gaze_2d_angles': gaze_2d_angles_np
                        }
                        self.samples.append(sample_data)

                    except ValueError as e:
                        print(f"Warning: ValueError parsing numeric data in line {line_idx+1} in {annotation_file}: '{line.strip()}'. Error: {e}. Skipping.")
                        continue
                    except IndexError as e:
                        print(f"Warning: IndexError processing line {line_idx+1} in {annotation_file}: '{line.strip()}'. Error: {e}. Skipping.")
                        continue

    def _normalize_gaze_to_head(self, gaze_cam_np: np.ndarray, head_rvec_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Rotate 3-D gaze direction from camera to head coordinate system."""
        if np.linalg.norm(gaze_cam_np) == 0:
            # Degenerate – return default forward direction.
            return np.array([0.0, 0.0, -1.0], dtype=np.float32), np.array([0.0, 0.0], dtype=np.float32)

        # Rotation matrix: head -> camera.  We need camera -> head, i.e. R^T.
        R_head2cam, _ = cv2.Rodrigues(head_rvec_np.astype(np.float32))
        R_cam2head = R_head2cam.T  # inverse because R is orthonormal

        gaze_head_np = R_cam2head @ gaze_cam_np.astype(np.float32)
        gaze_head_np = gaze_head_np / (np.linalg.norm(gaze_head_np) + 1e-6)

        pitch = np.arcsin(np.clip(-gaze_head_np[1], -1.0, 1.0))
        yaw = np.arctan2(-gaze_head_np[0], -gaze_head_np[2])
        gaze_angles_np = np.array([pitch, yaw], dtype=np.float32)
        
        return gaze_head_np.astype(np.float32), gaze_angles_np

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_path = sample['image_path']
        
        # Make a fresh copy of landmarks; gaze will be recomputed after head-pose normalisation
        landmarks = sample['facial_landmarks'].copy()

        image = None

        if self.use_cache and image_path in self.image_cache:
            image, landmarks = self.image_cache[image_path]
            landmarks = landmarks.copy() 
        else:
            image = Image.open(image_path)
            if self.input_channels == 1:
                image = image.convert('L')
            else:
                image = image.convert('RGB')

            # Crop to non-black region (content crop)
            image, landmarks = crop_to_content(image, landmarks, non_black_threshold=1)
            
            # Apply CLAHE 
            if self.use_clahe:
                image = apply_clahe(image) # Uses default clipLimit=2.0, tileGridSize=(8,8)

            if self.use_cache:
                self.image_cache[image_path] = (image.copy(), landmarks.copy())  # Store copies

        if image is None:
            raise RuntimeError(f"Image object is None for sample {index} ({image_path}).")
         
        effective_width, effective_height = image.size 

        # --- Head-pose normalisation (rotate gaze into head coordinate system) ---
        gaze_3d_direction_head, gaze_2d_angles = self._normalize_gaze_to_head(
            sample['gaze_direction_cam_3d'], sample['head_pose_rvec'])
        # gaze_3d_direction_head = sample['gaze_direction_cam_3d']
        # gaze_2d_angles = sample['gaze_2d_angles']

        # Affine Augmentation (if training)
        if self.is_train and self.affine_aug and random.random() > 0.5:
            image, landmarks, gaze_2d_angles, gaze_3d_direction_head = random_affine_with_landmarks(
                image, landmarks, gaze_2d_angles, gaze_3d_direction_head)

        # Horizontal Flip (if training)
        if self.is_train and self.flip_aug and random.random() > 0.5:
            image, landmarks, gaze_2d_angles, gaze_3d_direction_head = horizontal_flip(
                image, landmarks, gaze_2d_angles, effective_width, gaze_3d_direction_head)

        # Normalize landmarks to [0, 1]
        landmarks = normalize_landmarks(landmarks, effective_width, effective_height)

        # Apply label smoothing to landmarks (if training and enabled)
        if self.is_train and self.label_smoothing > 0:
            landmarks = landmarks_smoothing(landmarks, smoothing_factor=self.label_smoothing)

        # current_image is now a PIL Image in L mode (grayscale, equalized)
        item_to_return = {}
        if self.transform:
            item_to_return['image'] = self.transform(image)
        else:
            item_to_return['image'] = TF.to_tensor(image)

        # Convert numpy arrays to tensors
        item_to_return['gaze_screen_px'] = torch.from_numpy(sample['gaze_screen_px'])
        item_to_return['facial_landmarks'] = torch.from_numpy(landmarks.astype(np.float32))
        item_to_return['head_pose_rvec'] = torch.from_numpy(sample['head_pose_rvec'])
        item_to_return['head_pose_tvec'] = torch.from_numpy(sample['head_pose_tvec'])
        item_to_return['face_center_cam'] = torch.from_numpy(sample['face_center_cam'])
        item_to_return['gaze_target_cam'] = torch.from_numpy(sample['gaze_target_cam'])
        item_to_return['gaze_direction_cam_3d'] = torch.from_numpy(gaze_3d_direction_head.astype(np.float32))
        item_to_return['gaze_2d_angles'] = torch.from_numpy(gaze_2d_angles.astype(np.float32))
        
        # Bin encoding for yaw / pitch
        num_bins = self.num_angle_bins
        bin_width = self.angle_bin_width

        def _angle_to_bin(angle_rad: float) -> int:
            angle_deg = np.degrees(angle_rad)
            half_range = (num_bins * bin_width) / 2.0
            idx = int(np.floor((angle_deg + half_range) / bin_width))
            idx = np.clip(idx, 0, num_bins - 1)
            return idx

        pitch_bin_idx = _angle_to_bin(gaze_2d_angles[0])
        yaw_bin_idx = _angle_to_bin(gaze_2d_angles[1])

        # One-hot vectors
        pitch_onehot = np.zeros(num_bins, dtype=np.float32)
        yaw_onehot = np.zeros(num_bins, dtype=np.float32)
        pitch_onehot[pitch_bin_idx] = 1.0
        yaw_onehot[yaw_bin_idx] = 1.0

        item_to_return['pitch_bin_onehot'] = torch.from_numpy(pitch_onehot)
        item_to_return['yaw_bin_onehot'] = torch.from_numpy(yaw_onehot)
        
        # Non-tensor data
        item_to_return['eval_eye'] = sample['eval_eye'] # String
        item_to_return['image_path'] = image_path # String, for debugging or reference

        return item_to_return


class WFLWDataset(BaseDataset):
    """
    WFLW (Wider Facial Landmarks in the Wild) Dataset
    98 facial landmarks with attribute annotations

    Landmarks:
    - 60: Left outer eye
    - 64: Left inner eye  
    - 68: Right inner eye
    - 72: Right outer eye
    - 76: Left mouth
    - 82 Right mouth
    """
    def __init__(self, annotation_file, images_dir, transform=None, is_train=False, affine_aug=True, flip_aug=True, use_cache=False, label_smoothing=0.0, mpii_landmarks=False, input_channels=1, use_clahe=False):
        super().__init__(transform)
        self.annotation_file = annotation_file
        self.images_dir = images_dir
        self.samples = []
        self.is_train = is_train
        self.affine_aug = affine_aug
        self.flip_aug = flip_aug
        self.use_cache = use_cache
        self.label_smoothing = label_smoothing
        self.mpii_landmarks = mpii_landmarks
        self.input_channels = input_channels
        self.use_clahe = use_clahe

        if self.use_cache:
            self.image_cache = {}
            print("Image caching enabled for WFLWDataset.")
        
        self._load_annotations()

    def _load_annotations(self):
        if not os.path.exists(self.annotation_file):
            raise FileNotFoundError(f"Annotation file not found: {self.annotation_file}")

        with open(self.annotation_file, 'r') as f:
            for line_idx, line in enumerate(f):
                parts = line.strip().split(' ')
                if len(parts) != 207:  # 196 + 4 + 6 + 1
                    print(f"Warning: Malformed line {line_idx+1} in {self.annotation_file}. Expected 207 parts, got {len(parts)}. Skipping.")
                    continue

                try:
                    # 98 landmarks (196 coordinates)
                    landmarks_coords = [float(p) for p in parts[:196]]
                    landmarks = np.array(landmarks_coords, dtype=np.float32).reshape(98, 2)
                    
                    # Detection rectangle
                    bbox = np.array([float(p) for p in parts[196:200]], dtype=np.float32)
                    
                    # Attributes
                    pose = int(parts[200])
                    expression = int(parts[201])
                    illumination = int(parts[202])
                    makeup = int(parts[203])
                    occlusion = int(parts[204])
                    blur = int(parts[205])
                    
                    # Image name
                    image_name = parts[206]
                    image_path = os.path.join(self.images_dir, image_name)
                    
                    if not os.path.exists(image_path):
                        print(f"Warning: Image not found {image_path}. Skipping.")
                        continue

                    sample_data = {
                        'image_path': image_path,
                        'landmarks': landmarks,
                        'bbox': bbox,
                        'pose': pose,
                        'expression': expression,
                        'illumination': illumination,
                        'makeup': makeup,
                        'occlusion': occlusion,
                        'blur': blur,
                        'image_name': image_name
                    }
                    self.samples.append(sample_data)

                except (ValueError, IndexError) as e:
                    print(f"Warning: Error parsing line {line_idx+1} in {self.annotation_file}: {e}. Skipping.")
                    continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_path = sample['image_path']
        
        landmarks = sample['landmarks'].copy()
        bbox = sample['bbox'].copy()

        if self.use_cache and image_path in self.image_cache:
            image, landmarks = self.image_cache[image_path]
            landmarks = landmarks.copy()
        else:
            image = Image.open(image_path)
            if self.input_channels == 1:
                image = image.convert('L')
            else:
                image = image.convert('RGB')
            
            # Crop based on bounding box
            x_min, y_min, x_max, y_max = bbox
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
            
            # Ensure bbox is within image bounds
            img_width, img_height = image.size
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_width, x_max)
            y_max = min(img_height, y_max)
            
            # Crop image
            image = image.crop((x_min, y_min, x_max, y_max))
            
            # Adjust landmarks relative to cropped region
            landmarks[:, 0] -= x_min
            landmarks[:, 1] -= y_min
            
            # Apply CLAHE
            if self.use_clahe:
                image = apply_clahe(image)

            if self.use_cache:
                self.image_cache[image_path] = (image.copy(), landmarks.copy())

        if image is None:
            raise RuntimeError(f"Image object is None for sample {index} ({image_path}).")
         
        effective_width, effective_height = image.size 

        # Affine Augmentation
        if self.is_train and self.affine_aug and random.random() > 0.5:
            image, landmarks = random_affine_with_landmarks(image, landmarks)

        # Horizontal Flip
        if self.is_train and self.flip_aug and random.random() > 0.5:
            image, landmarks = horizontal_flip(image, landmarks, effective_width)
            # WFLW landmark flipping indices (symmetric pairs)
            flip_indices = [32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
                           0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                           46, 45, 44, 43, 42, 50, 49, 48, 47, 37, 36, 35, 34, 33, 41, 40, 39, 38,
                           51, 52, 53, 54, 59, 58, 57, 56, 55, 72, 71, 70, 69, 68, 75, 74, 73, 60,
                           61, 63, 62, 67, 66, 65, 64, 82, 81, 80, 79, 78, 77, 76, 87, 86, 85, 84, 83,
                           92, 91, 90, 89, 88, 95, 94, 93, 97, 96]
            landmarks = landmarks[flip_indices, :]

        # Normalize landmarks
        landmarks = normalize_landmarks(landmarks, effective_width, effective_height)

        # Apply label smoothing
        if self.is_train and self.label_smoothing > 0:
            landmarks = landmarks_smoothing(landmarks, smoothing_factor=self.label_smoothing)

        # Extract MPII-style landmarks if requested
        if self.mpii_landmarks:
            landmarks = landmarks[[60, 64, 68, 72, 76, 82], :]

        item_to_return = {}
        if self.transform:
            item_to_return['image'] = self.transform(image)
        else:
            item_to_return['image'] = TF.to_tensor(image)

        # Convert to tensors
        item_to_return['landmarks'] = torch.from_numpy(landmarks.astype(np.float32))
        item_to_return['bbox'] = torch.from_numpy(sample['bbox'])
        item_to_return['pose'] = torch.tensor(sample['pose'], dtype=torch.long)
        item_to_return['expression'] = torch.tensor(sample['expression'], dtype=torch.long)
        item_to_return['illumination'] = torch.tensor(sample['illumination'], dtype=torch.long)
        item_to_return['makeup'] = torch.tensor(sample['makeup'], dtype=torch.long)
        item_to_return['occlusion'] = torch.tensor(sample['occlusion'], dtype=torch.long)
        item_to_return['blur'] = torch.tensor(sample['blur'], dtype=torch.long)
        
        item_to_return['image_name'] = sample['image_name']
        item_to_return['image_path'] = image_path

        return item_to_return


class Face300WDataset(BaseDataset):
    """
    300W (300 Faces In-the-Wild) Dataset
    68 facial landmarks

    Landmarks:
    - 36: Left outer eye
    - 39: Left inner eye
    - 42: Right inner eye
    - 45: Right outer eye
    - 48: Left mouth
    - 54: Right mouth
    """
    def __init__(self, root_dir, transform=None, is_train=False, affine_aug=True, flip_aug=True, use_cache=False, label_smoothing=0.0, subset=None, mpii_landmarks=False, padding_ratio=0.3, translation_ratio=0.2, train_test_split=0.8, split='train', split_seed=42, input_channels=1, use_clahe=False):
        super().__init__(transform)
        self.root_dir = root_dir
        self.samples = []
        self.is_train = is_train
        self.affine_aug = affine_aug
        self.flip_aug = flip_aug
        self.use_cache = use_cache
        self.label_smoothing = label_smoothing
        self.subset = subset  # Can be 'indoor', 'outdoor', or None for both
        self.mpii_landmarks = mpii_landmarks
        self.padding_ratio = padding_ratio
        self.translation_ratio = translation_ratio
        self.train_test_split = train_test_split  # Ratio of training data (0.0 to 1.0)
        self.split = split  # 'train' or 'test'
        self.split_seed = split_seed  # Seed for reproducible splits
        self.input_channels = input_channels
        self.use_clahe = use_clahe
        
        if self.use_cache:
            self.image_cache = {}
            print("Image caching enabled for Face300WDataset.")
        
        self._load_annotations()

    def _load_annotations(self):
        # Define subdirectories to process
        subdirs = []
        if self.subset == 'indoor':
            subdirs = ['01_Indoor']
        elif self.subset == 'outdoor':
            subdirs = ['02_Outdoor']
        else:
            subdirs = ['01_Indoor', '02_Outdoor']

        all_samples = []  # Collect all samples first, then split
        
        for subdir in subdirs:
            subdir_path = os.path.join(self.root_dir, subdir)
            if not os.path.exists(subdir_path):
                print(f"Warning: Subdirectory not found {subdir_path}. Skipping.")
                continue

            # Find all .pts files in the subdirectory
            pts_files = [f for f in os.listdir(subdir_path) if f.endswith('.pts')]
            
            for pts_file in pts_files:
                pts_path = os.path.join(subdir_path, pts_file)
                
                # Corresponding image file (.png)
                image_file = pts_file.replace('.pts', '.png')
                image_path = os.path.join(subdir_path, image_file)
                
                if not os.path.exists(image_path):
                    print(f"Warning: Image file not found {image_path}. Skipping.")
                    continue

                try:
                    landmarks = self._parse_pts_file(pts_path)
                    if landmarks is None:
                        continue

                    sample_data = {
                        'image_path': image_path,
                        'landmarks': landmarks,
                        'subset': subdir,
                        'image_name': image_file
                    }
                    all_samples.append(sample_data)

                except Exception as e:
                    print(f"Warning: Error processing {pts_path}: {e}. Skipping.")
                    continue

        print(f"Loaded {len(all_samples)} total samples from 300W dataset.")
        
        # Perform train/test split
        if len(all_samples) == 0:
            print("Warning: No samples found to split.")
            self.samples = []
            return
            
        # Set random seed for reproducible splits
        import random
        random.seed(self.split_seed)
        
        # Shuffle samples to ensure random distribution
        all_samples_copy = all_samples.copy()
        random.shuffle(all_samples_copy)
        
        # Calculate split index
        n_total = len(all_samples_copy)
        n_train = int(n_total * self.train_test_split)
        
        # Split samples
        if self.split == 'train':
            self.samples = all_samples_copy[:n_train]
            print(f"Using {len(self.samples)} samples for training (split ratio: {self.train_test_split:.2f})")
        elif self.split == 'test':
            self.samples = all_samples_copy[n_train:]
            print(f"Using {len(self.samples)} samples for testing (split ratio: {1-self.train_test_split:.2f})")
        else:
            raise ValueError(f"Invalid split value: {self.split}. Must be 'train' or 'test'.")
            
        if len(self.samples) == 0:
            print(f"Warning: No samples in {self.split} split. Consider adjusting train_test_split ratio.")
        
        # Reset random seed to avoid affecting other random operations
        random.seed()

    def _parse_pts_file(self, pts_path):
        """Parse a .pts annotation file and return landmarks as numpy array."""
        try:
            with open(pts_path, 'r') as f:
                lines = f.readlines()

            # Find the line with n_points
            n_points = None
            start_idx = None
            
            for i, line in enumerate(lines):
                line = line.strip()
                if line.startswith('n_points:'):
                    n_points = int(line.split(':')[1].strip())
                elif line == '{':
                    start_idx = i + 1
                    break

            if n_points is None or start_idx is None:
                print(f"Warning: Could not parse header in {pts_path}")
                return None

            if n_points != 68:
                print(f"Warning: Expected 68 points, got {n_points} in {pts_path}")
                return None

            # Read landmark coordinates
            landmarks = []
            for i in range(start_idx, start_idx + n_points):
                if i >= len(lines):
                    print(f"Warning: Not enough coordinate lines in {pts_path}")
                    return None
                
                line = lines[i].strip()
                if line == '}':
                    break
                
                coords = line.split()
                if len(coords) != 2:
                    print(f"Warning: Invalid coordinate format in {pts_path}: {line}")
                    return None
                
                x, y = float(coords[0]), float(coords[1])
                landmarks.append([x, y])

            if len(landmarks) != n_points:
                print(f"Warning: Expected {n_points} landmarks, got {len(landmarks)} in {pts_path}")
                return None

            return np.array(landmarks, dtype=np.float32)

        except Exception as e:
            print(f"Error parsing {pts_path}: {e}")
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_path = sample['image_path']
        
        landmarks = sample['landmarks'].copy()

        if self.use_cache and image_path in self.image_cache:
            image, landmarks = self.image_cache[image_path]
            landmarks = landmarks.copy()
        else:
            image = Image.open(image_path)
            if self.input_channels == 1:
                image = image.convert('L')
            else:
                image = image.convert('RGB')

            # Crop to face region based on landmarks with padding and optional random translation
            translation_ratio = self.translation_ratio if self.is_train else 0.0
            image, landmarks = crop_to_landmarks(image, landmarks, 
                                               padding_ratio=self.padding_ratio,
                                               translation_ratio=translation_ratio)
            
            # Apply CLAHE
            if self.use_clahe:
                image = apply_clahe(image)

            if self.use_cache:
                self.image_cache[image_path] = (image.copy(), landmarks.copy())

        if image is None:
            raise RuntimeError(f"Image object is None for sample {index} ({image_path}).")
         
        effective_width, effective_height = image.size 

        # Affine Augmentation
        if self.is_train and self.affine_aug and random.random() > 0.5:
            image, landmarks = random_affine_with_landmarks(image, landmarks)

        # Horizontal Flip
        if self.is_train and self.flip_aug and random.random() > 0.5:
            image, landmarks = horizontal_flip(image, landmarks, effective_width)
            # 68-point landmark flipping indices (iBUG 68-point standard)
            flip_indices = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                           26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
                           27, 28, 29, 30,
                           35, 34, 33, 32, 31,
                           45, 44, 43, 42, 47, 46, 39, 38, 37, 36, 41, 40,
                           54, 53, 52, 51, 50, 49, 48,
                           59, 58, 57, 56, 55, 64, 63, 62, 61, 60, 67, 66, 65]
            landmarks = landmarks[flip_indices, :]

        # Normalize landmarks
        landmarks = normalize_landmarks(landmarks, effective_width, effective_height)

        # Apply label smoothing
        if self.is_train and self.label_smoothing > 0:
            landmarks = landmarks_smoothing(landmarks, smoothing_factor=self.label_smoothing)

        # Extract MPII-style landmarks if requested
        if self.mpii_landmarks:
            landmarks = landmarks[[36, 39, 42, 45, 48, 54], :]

        item_to_return = {}
        if self.transform:
            item_to_return['image'] = self.transform(image)
        else:
            item_to_return['image'] = TF.to_tensor(image)

        # Convert to tensors
        item_to_return['landmarks'] = torch.from_numpy(landmarks.astype(np.float32))
        item_to_return['subset'] = sample['subset']
        item_to_return['image_name'] = sample['image_name']
        item_to_return['image_path'] = image_path

        return item_to_return


class MPIIFaceGazeMatDataset(BaseDataset):
    """Dataset class for MPIIFaceGaze normalized .mat files."""
    def __init__(self,
                 dataset_path: str,
                 participant_ids: list[int] | None = None,
                 transform=None,
                 input_channels: int = 3,
                 use_cache: bool = False,
                 use_clahe: bool = False,
                 downscale_size: int | tuple[int, int] | None = 224,
                 affine_aug: bool = False,
                 horizontal_flip: bool = False,
                 angle_bin_width: float = 3.0,
                 num_angle_bins: int = 14,
                 label_smoothing: float = 0.1):
        super().__init__(transform)
        self.dataset_path = dataset_path
        self.input_channels = input_channels
        self.use_cache = use_cache
        self.use_clahe = use_clahe
        self.downscale_size = downscale_size
        self.affine_aug = affine_aug
        self.horizontal_flip = horizontal_flip
        self.angle_bin_width = angle_bin_width
        self.num_angle_bins = num_angle_bins
        self.label_smoothing = label_smoothing
        
        if participant_ids is None:
            participant_ids = list(range(15))

        # Hold file-level metadata so that we can locate a sample quickly.
        self.files = []
        cumulative_samples = 0
        for pid in participant_ids:
            mat_path = os.path.join(self.dataset_path, f"p{pid:02d}.mat")
            if not os.path.isfile(mat_path):
                print(f"Warning: MAT file not found: {mat_path}. Skipping participant {pid}.")
                continue
            with h5py.File(mat_path, 'r') as f:
                if 'Data' not in f or 'label' not in f['Data']:
                    print(f"Warning: Unexpected MAT structure in {mat_path}. Skipping.")
                    continue
                n_samples = f['Data']['label'].shape[0]

            self.files.append({'path': mat_path, 'n_samples': n_samples, 'cumulative_start': cumulative_samples})
            cumulative_samples += n_samples

        self.total_samples = cumulative_samples
        if self.total_samples == 0:
            raise RuntimeError("No samples found in the provided participant list and dataset path.")

        # Workaround for compatibility with the training code
        self.samples = list(range(self.total_samples))

        # Pre-compute cumulative starts to locate a sample with bisect.
        self._cumulative_starts = [info['cumulative_start'] for info in self.files] + [self.total_samples]

        # Optional in-memory cache for PIL images keyed by (file_path, local_idx)
        if self.use_cache:
            self.image_cache: dict[tuple[str, int], Image.Image] = {}
            print("Image caching enabled for MPIIFaceGazeMatDataset.")
            self._preload_cache()
            print(f"Pre-loaded {len(self.image_cache)} images into cache.")

    def _preload_cache(self):
        """Pre-load all images and labels into cache during initialization."""
        for global_idx in tqdm(range(self.total_samples), desc="Loading images to cache"):
            file_info, local_idx = self._locate_sample(global_idx)
            cache_key = (file_info['path'], local_idx)

            # Load raw data
            img_pil, gaze_2d_angles_np, head_pose_angles_np, landmarks_np = self._load_from_mat(
                file_info['path'], local_idx)

            # Grayscale conversion (optional)
            if self.input_channels == 1 and img_pil.mode != 'L':
                img_pil = img_pil.convert('L')
            elif self.input_channels == 3 and img_pil.mode != 'RGB':
                img_pil = img_pil.convert('RGB')

            # CLAHE (optional)
            if self.use_clahe:
                img_pil = apply_clahe(img_pil)

            # Down-scaling
            effective_width, effective_height = img_pil.size
            if self.downscale_size is not None:
                if isinstance(self.downscale_size, int):
                    target_size = (self.downscale_size, self.downscale_size)
                else:
                    target_size = self.downscale_size
                scale_x = target_size[1] / effective_width
                scale_y = target_size[0] / effective_height
                img_pil = img_pil.resize(target_size, Image.BILINEAR)
                landmarks_np[:, 0] *= scale_x
                landmarks_np[:, 1] *= scale_y
                effective_width, effective_height = target_size[1], target_size[0]

            # Store in cache
            self.image_cache[cache_key] = (
                img_pil.copy(),
                gaze_2d_angles_np.copy(),
                head_pose_angles_np.copy(),
                landmarks_np.copy(),
            )

    def _locate_sample(self, global_idx: int) -> tuple[dict, int]:
        """Return (file_info_dict, local_idx) for the *global* dataset index."""
        file_idx = bisect.bisect_right(self._cumulative_starts, global_idx) - 1
        file_info = self.files[file_idx]
        local_idx = global_idx - file_info['cumulative_start']

        return file_info, local_idx 

    def _load_from_mat(self, file_path: str, local_idx: int):
        """Read image and label for local_idx from file_path."""
        with h5py.File(file_path, 'r') as f:
            data_grp = f['Data']
            # Load image
            img_np = data_grp['data'][local_idx]  # shape (3, H, W) – BGR, uint8
            img_np = np.transpose(img_np, (1, 2, 0))  # (H, W, C)
            # BGR to RGB
            img_np = img_np[:, :, ::-1]
            # Horizontal flip
            img_np = np.fliplr(img_np)
            # Convert to PIL
            img_pil = Image.fromarray(img_np.astype(np.uint8))

            # Load labels
            label_np = data_grp['label'][local_idx].astype(np.float32)  # (16,)
            gaze_2d_angles_np = label_np[0:2]  # pitch, yaw (radians)
            head_pose_angles_np = label_np[2:4]
            landmarks_flat = label_np[4:16]  # 12 values
            landmarks_np = landmarks_flat.reshape(6, 2)  # (6,2) in pixel coords
            # Update landmarks x after horizontal flip (448px image width)
            landmarks_np[:, 0] = 448.0 - landmarks_np[:, 0]

            # Flip yaw sign due to horizontal flip
            gaze_2d_angles_np[1] *= -1.0
            head_pose_angles_np[1] *= -1.0

        return img_pil, gaze_2d_angles_np, head_pose_angles_np, landmarks_np
    
    def _angle_to_bin(self, angle_rad: float) -> int:
        angle_deg = np.degrees(angle_rad)
        half_range = (self.num_angle_bins * self.angle_bin_width) / 2.0
        idx = int(np.floor((angle_deg + half_range) / self.angle_bin_width))
        idx = np.clip(idx, 0, self.num_angle_bins - 1)

        return idx

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):
        # Locate sample
        file_info, local_idx = self._locate_sample(index)
        cache_key = (file_info['path'], local_idx)
        if self.use_cache:
            # All images are pre-cached during initialization
            if cache_key not in self.image_cache:
                raise RuntimeError(f"Cache miss for pre-loaded data at index {index}. This should not happen.")
            
            # Retrieve fully-processed data
            img_pil, gaze_2d_angles_np, head_pose_angles_np, landmarks_np = self.image_cache[cache_key]

            gaze_2d_angles_np = gaze_2d_angles_np.copy()
            head_pose_angles_np = head_pose_angles_np.copy()
            landmarks_np = landmarks_np.copy()
        else:
            # Load raw data
            img_pil, gaze_2d_angles_np, head_pose_angles_np, landmarks_np = self._load_from_mat(
                file_info['path'], local_idx)

            # CLAHE (optional)
            if self.use_clahe:
                img_pil = apply_clahe(img_pil)

            # Grayscale conversion (optional)
            if self.input_channels == 1 and img_pil.mode != 'L':
                img_pil = img_pil.convert('L')
            elif self.input_channels == 3 and img_pil.mode != 'RGB':
                img_pil = img_pil.convert('RGB')

            # Down-scaling (optional)
            effective_width, effective_height = img_pil.size
            if self.downscale_size is not None:
                if isinstance(self.downscale_size, int):
                    target_size = (self.downscale_size, self.downscale_size)
                else:
                    target_size = self.downscale_size  # (H, W)
                scale_x = target_size[1] / effective_width
                scale_y = target_size[0] / effective_height
                img_pil = img_pil.resize(target_size, Image.BILINEAR)
                # Adjust landmarks to new resolution
                landmarks_np[:, 0] *= scale_x
                landmarks_np[:, 1] *= scale_y
                effective_width, effective_height = target_size[1], target_size[0]

        # Affine Augmentation
        if self.affine_aug and random.random() > 0.5:
            img_pil, landmarks_np, gaze_2d_angles_np, _ = random_affine_with_landmarks(
                img_pil, 
                landmarks_np, 
                gaze_2d_angles_np,
                degrees=(-10, 10),
                translate_fractions=(0.1, 0.1),
                scale_range=(0.8, 1.2)
            )

        # Horizontal flip (Optional)
        if self.horizontal_flip and random.random() > 0.5:
            img_pil, landmarks_np, gaze_2d_angles_np, _ = horizontal_flip(img_pil, landmarks_np, gaze_2d_angles_np, img_pil.size[0])

        # Normalize landmarks
        landmarks_np = normalize_landmarks(landmarks_np, effective_width, effective_height)

        # Apply label smoothing
        if self.label_smoothing > 0:
            landmarks_np = landmarks_smoothing(landmarks_np, smoothing_factor=self.label_smoothing)

        # Apply transform
        item = {}
        if self.transform:
            item['image'] = self.transform(img_pil)
        else:
            item['image'] = TF.to_tensor(img_pil)

        # Numerical labels
        item['gaze_2d_angles'] = torch.from_numpy(gaze_2d_angles_np.astype(np.float32))
        item['head_pose_angles'] = torch.from_numpy(head_pose_angles_np.astype(np.float32))
        item['facial_landmarks'] = torch.from_numpy(landmarks_np.astype(np.float32))

        # Bin encoding for yaw / pitch
        pitch_bin_idx = self._angle_to_bin(gaze_2d_angles_np[0])
        yaw_bin_idx = self._angle_to_bin(gaze_2d_angles_np[1])
        pitch_onehot = np.zeros(self.num_angle_bins, dtype=np.float32)
        yaw_onehot = np.zeros(self.num_angle_bins, dtype=np.float32)
        pitch_onehot[pitch_bin_idx] = 1.0
        yaw_onehot[yaw_bin_idx] = 1.0
        item['pitch_bin_onehot'] = torch.from_numpy(pitch_onehot)
        item['yaw_bin_onehot'] = torch.from_numpy(yaw_onehot)

        # Meta-information
        item['file_path'] = file_info['path']
        item['local_idx'] = local_idx

        return item
    