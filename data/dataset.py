from torch.utils.data import Dataset
import os
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import random

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
    def __init__(self, dataset_path, participant_ids, transform=None, img_size=224, is_train=False):
        super().__init__(transform)
        self.dataset_path = dataset_path
        self.samples = []
        self.img_size = img_size
        self.is_train = is_train

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
        landmarks_np = sample['facial_landmarks'].copy()
        
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: File not found {image_path} for sample index {index}. Attempting to load next valid sample.")
            if len(self.samples) == 0:
                 raise RuntimeError("No samples available in the dataset.")
            if len(self.samples) > 1: # Avoid infinite loop if only one sample and it's missing
                 return self.__getitem__((index + 1) % len(self.samples)) # Try next, wrap around
            else: # Only one sample and it's missing
                 raise RuntimeError(f"Could not load the only image in the dataset: {image_path}")

        # Image and Landmark Augmentation (if training)
        if self.is_train and random.random() > 0.5: # 50% chance to flip
            image = TF.hflip(image)
            landmarks_np[:, 0] = self.img_size - landmarks_np[:, 0]

        # Normalize landmarks to [0, 1]
        landmarks_np[:, 0] /= self.img_size
        landmarks_np[:, 1] /= self.img_size
        # Clip to ensure they are within [0,1] after division and potential flipping arithmetic
        landmarks_np = np.clip(landmarks_np, 0.0, 1.0)

        item_to_return = {}
        if self.transform:
            item_to_return['image'] = self.transform(image)
        else:
            item_to_return['image'] = TF.to_tensor(image)

        # Convert numpy arrays to tensors
        item_to_return['gaze_screen_px'] = torch.from_numpy(sample['gaze_screen_px'])
        item_to_return['facial_landmarks'] = torch.from_numpy(landmarks_np.astype(np.float32))
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

