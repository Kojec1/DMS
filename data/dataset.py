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
    def __init__(self, dataset_path, participant_ids, transform=None, is_train=False, crop_to_face_bbox=True, train_crop_padding_range=(1.0, 2.0), test_crop_padding_factor=1.0, train_random_shift_fraction=0.2):
        super().__init__(transform)
        self.dataset_path = dataset_path
        self.samples = []
        self.is_train = is_train
        self.crop_to_face_bbox = crop_to_face_bbox
        if not isinstance(train_crop_padding_range, tuple) or len(train_crop_padding_range) != 2:
            raise ValueError("train_crop_padding_range must be a tuple of two floats (min, max).")
        if not (isinstance(test_crop_padding_factor, float) or isinstance(test_crop_padding_factor, int)):
            raise ValueError("test_crop_padding_factor must be a float or int.")
        if not (isinstance(train_random_shift_fraction, float) or isinstance(train_random_shift_fraction, int)):
            raise ValueError("train_random_shift_fraction must be a float or int.")

        self.train_crop_padding_range = train_crop_padding_range
        self.test_crop_padding_factor = float(test_crop_padding_factor)
        self.train_random_shift_fraction = float(train_random_shift_fraction)
        
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
        # original_landmarks are in the coordinate system of the original, uncropped image
        original_landmarks_np = sample['facial_landmarks'].copy() 
        
        try:
            image = Image.open(image_path).convert('RGB')
            original_width, original_height = image.size
        except FileNotFoundError:
            print(f"Warning: File not found {image_path} for sample index {index}. Attempting to load next valid sample.")
            if len(self.samples) == 0:
                 raise RuntimeError("No samples available in the dataset.")
            if len(self.samples) > 1: # Avoid infinite loop if only one sample and it's missing
                 return self.__getitem__((index + 1) % len(self.samples)) # Try next, wrap around
            else: # Only one sample and it's missing
                 raise RuntimeError(f"Could not load the only image in the dataset: {image_path}")

        current_image = image
        # current_landmarks will be adjusted if cropping occurs
        current_landmarks_np = original_landmarks_np.copy()
        effective_width, effective_height = original_width, original_height

        if self.crop_to_face_bbox:
            padding_factor_to_use = 0.0
            apply_random_shift = False

            if self.is_train:
                padding_factor_to_use = random.uniform(self.train_crop_padding_range[0], self.train_crop_padding_range[1])
                if self.train_random_shift_fraction > 0.0:
                    apply_random_shift = True
            else: # is_test
                padding_factor_to_use = self.test_crop_padding_factor

            # Calculate bounding box from original landmarks
            x_min_lm, y_min_lm = np.min(original_landmarks_np, axis=0)
            x_max_lm, y_max_lm = np.max(original_landmarks_np, axis=0)

            bbox_width = x_max_lm - x_min_lm
            bbox_height = y_max_lm - y_min_lm
            
            # Ensure non-negative bbox dimensions for padding calculation
            bbox_width = max(0, bbox_width)
            bbox_height = max(0, bbox_height)

            pad_w_half = (bbox_width * padding_factor_to_use) / 2.0
            pad_h_half = (bbox_height * padding_factor_to_use) / 2.0

            # Calculate initial floating point padded crop box (before shift)
            f_padded_crop_x1 = x_min_lm - pad_w_half
            f_padded_crop_y1 = y_min_lm - pad_h_half
            f_padded_crop_x2 = x_max_lm + pad_w_half
            f_padded_crop_y2 = y_max_lm + pad_h_half
            
            # Initialize final floating crop coordinates with the padded box
            final_f_crop_x1, final_f_crop_y1 = f_padded_crop_x1, f_padded_crop_y1
            final_f_crop_x2, final_f_crop_y2 = f_padded_crop_x2, f_padded_crop_y2

            if apply_random_shift:
                # Max shift to keep original landmark bbox inside the padded box
                # dx_max_lm_constraint is pad_w_half, dx_min_lm_constraint is -pad_w_half
                dx_min_lm_constraint = -pad_w_half 
                dx_max_lm_constraint = pad_w_half
                dy_min_lm_constraint = -pad_h_half
                dy_max_lm_constraint = pad_h_half
                
                # Constraint from image boundaries for the shifted box:
                # 0 <= f_padded_crop_x1 + dx  => dx >= -f_padded_crop_x1
                # f_padded_crop_x2 + dx <= original_width => dx <= original_width - f_padded_crop_x2
                dx_min_img_constraint = -f_padded_crop_x1
                dx_max_img_constraint = original_width - f_padded_crop_x2
                dy_min_img_constraint = -f_padded_crop_y1
                dy_max_img_constraint = original_height - f_padded_crop_y2

                # Combine landmark and image boundary constraints for dx
                actual_dx_min = max(dx_min_lm_constraint, dx_min_img_constraint)
                actual_dx_max = min(dx_max_lm_constraint, dx_max_img_constraint)
                
                # Combine landmark and image boundary constraints for dy
                actual_dy_min = max(dy_min_lm_constraint, dy_min_img_constraint)
                actual_dy_max = min(dy_max_lm_constraint, dy_max_img_constraint)

                random_dx = 0.0
                random_dy = 0.0

                if actual_dx_max > actual_dx_min: # Check if a valid range for dx exists
                    padded_box_width = f_padded_crop_x2 - f_padded_crop_x1
                    # Avoid issues if padded_box_width is zero or very small
                    if padded_box_width > 1e-6 : 
                        max_abs_shift_x = self.train_random_shift_fraction * padded_box_width
                        
                        # Intersect with user-defined relative shift limit
                        shift_limit_x_min = max(actual_dx_min, -max_abs_shift_x)
                        shift_limit_x_max = min(actual_dx_max, max_abs_shift_x)

                        if shift_limit_x_max > shift_limit_x_min:
                             random_dx = random.uniform(shift_limit_x_min, shift_limit_x_max)
                
                if actual_dy_max > actual_dy_min: # Check if a valid range for dy exists
                    padded_box_height = f_padded_crop_y2 - f_padded_crop_y1
                    if padded_box_height > 1e-6:
                        max_abs_shift_y = self.train_random_shift_fraction * padded_box_height

                        # Intersect with user-defined relative shift limit
                        shift_limit_y_min = max(actual_dy_min, -max_abs_shift_y)
                        shift_limit_y_max = min(actual_dy_max, max_abs_shift_y)

                        if shift_limit_y_max > shift_limit_y_min:
                            random_dy = random.uniform(shift_limit_y_min, shift_limit_y_max)
                
                final_f_crop_x1 += random_dx
                final_f_crop_y1 += random_dy
                final_f_crop_x2 += random_dx
                final_f_crop_y2 += random_dy
            
            # Convert to integer coordinates for cropping, clamping to original image boundaries
            crop_x1_final = int(round(np.maximum(0, final_f_crop_x1)))
            crop_y1_final = int(round(np.maximum(0, final_f_crop_y1)))
            crop_x2_final = int(round(np.minimum(original_width, final_f_crop_x2)))
            crop_y2_final = int(round(np.minimum(original_height, final_f_crop_y2)))

            if crop_x1_final < crop_x2_final and crop_y1_final < crop_y2_final:
                cropped_image_candidate = image.crop((crop_x1_final, crop_y1_final, crop_x2_final, crop_y2_final))
                if cropped_image_candidate.width > 0 and cropped_image_candidate.height > 0:
                    current_image = cropped_image_candidate
                    effective_width, effective_height = current_image.size
                    
                    # Adjust landmarks to the new cropped image coordinate system
                    current_landmarks_np[:, 0] -= crop_x1_final
                    current_landmarks_np[:, 1] -= crop_y1_final
                else:
                    print(f"Warning: Crop for {image_path} resulted in zero dimension. Original image used. Box:({crop_x1_final},{crop_y1_final},{crop_x2_final},{crop_y2_final})")
            else:
                print(f"Warning: Invalid crop box calculated for {image_path}. Original image used. Box:({crop_x1_final},{crop_y1_final},{crop_x2_final},{crop_y2_final})")

        # Image and Landmark Augmentation (if training)
        if self.is_train and random.random() > 0.5: # 50% chance to flip
            current_image = TF.hflip(current_image)
            # Adjust landmarks for flip, relative to the effective_width (which is cropped width if crop happened)
            current_landmarks_np[:, 0] = effective_width - current_landmarks_np[:, 0]

        # Normalize landmarks to [0, 1] using effective_width and effective_height
        if effective_width > 0:
            current_landmarks_np[:, 0] /= effective_width
        else:
            current_landmarks_np[:, 0] = 0.0 # Avoid division by zero; set to 0 if width is 0
        
        if effective_height > 0:
            current_landmarks_np[:, 1] /= effective_height
        else:
            current_landmarks_np[:, 1] = 0.0 # Avoid division by zero; set to 0 if height is 0
            
        # Clip to ensure they are within [0,1] after division and potential flipping arithmetic
        current_landmarks_np = np.clip(current_landmarks_np, 0.0, 1.0)

        item_to_return = {}
        if self.transform:
            item_to_return['image'] = self.transform(current_image)
        else:
            item_to_return['image'] = TF.to_tensor(current_image)

        # Convert numpy arrays to tensors
        item_to_return['gaze_screen_px'] = torch.from_numpy(sample['gaze_screen_px'])
        item_to_return['facial_landmarks'] = torch.from_numpy(current_landmarks_np.astype(np.float32))
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

