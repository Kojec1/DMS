from torch.utils.data import Dataset
import os
import numpy as np
import torch
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF
import random
from .augmentation import horizontal_flip, normalize_landmarks, crop_to_content, apply_clahe, random_affine_with_landmarks, landmarks_smoothing

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
    def __init__(self, dataset_path, participant_ids, transform=None, is_train=False, affine_aug=True, flip_aug=True, use_cache=False, label_smoothing=0.0):
        super().__init__(transform)
        self.dataset_path = dataset_path
        self.samples = []
        self.is_train = is_train
        self.affine_aug = affine_aug
        self.flip_aug = flip_aug
        self.use_cache = use_cache
        self.label_smoothing = label_smoothing 
        
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
                            pitch = 0.0
                            yaw = 0.0
                            # normalized_gaze_vec = np.array([0,0,-1], dtype=np.float32) # Default forward gaze if needed elsewhere
                        else:
                            normalized_gaze_vec = gaze_direction_cam_3d / norm_gaze_direction_val
                            # Pitch (vertical angle)
                            # Clamp asin argument to [-1, 1] to prevent domain errors from float precision issues
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
                            'gaze_direction_cam_3d': gaze_direction_cam_3d.astype(np.float32),
                            'gaze_2d_angles': gaze_2d_angles_np
                        }
                        self.samples.append(sample_data)

                    except ValueError as e:
                        print(f"Warning: ValueError parsing numeric data in line {line_idx+1} in {annotation_file}: '{line.strip()}'. Error: {e}. Skipping.")
                        continue
                    except IndexError as e:
                        print(f"Warning: IndexError processing line {line_idx+1} in {annotation_file}: '{line.strip()}'. Error: {e}. Skipping.")
                        continue


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_path = sample['image_path']
        
        # Make a fresh copy of landmarks from metadata each time, as they might be modified by augmentations
        landmarks = sample['facial_landmarks'].copy()

        image = None

        if self.use_cache and image_path in self.image_cache:
            image, landmarks = self.image_cache[image_path]
            landmarks = landmarks.copy() 
        else:
            image = Image.open(image_path).convert('L') # Load and convert to grayscale

            # Crop to non-black region (content crop)
            image, landmarks = crop_to_content(image, landmarks, non_black_threshold=1)
            
            # Apply CLAHE 
            image = apply_clahe(image) # Uses default clipLimit=2.0, tileGridSize=(8,8)

            if self.use_cache:
                self.image_cache[image_path] = (image.copy(), landmarks.copy())  # Store copies

        if image is None:
            raise RuntimeError(f"Image object is None for sample {index} ({image_path}).")
         
        effective_width, effective_height = image.size 

        # Affine Augmentation (if training)
        if self.is_train and self.affine_aug and random.random() > 0.5:
            image, landmarks = random_affine_with_landmarks(image, landmarks)

        # Horizontal Flip (if training)
        if self.is_train and self.flip_aug and random.random() > 0.5:
            image, landmarks = horizontal_flip(image, landmarks, effective_width)
            # Swap landmark indices after horizontal flip
            # Original: 0:L_outer, 1:L_inner, 2:R_inner, 3:R_outer, 4:L_mouth, 5:R_mouth
            # Flipped:  0:R_outer, 1:R_inner, 2:L_inner, 3:L_outer, 4:R_mouth, 5:L_mouth
            landmarks = landmarks[[3, 2, 1, 0, 5, 4], :]

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
        item_to_return['gaze_direction_cam_3d'] = torch.from_numpy(sample['gaze_direction_cam_3d'])
        item_to_return['gaze_2d_angles'] = torch.from_numpy(sample['gaze_2d_angles'])
        
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
    def __init__(self, annotation_file, images_dir, transform=None, is_train=False, affine_aug=True, flip_aug=True, use_cache=False, label_smoothing=0.0, mpii_landmarks=False):
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
            image = Image.open(image_path).convert('L')
            
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
