import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import random
import cv2
import math


def horizontal_flip(image, landmarks_np, gaze_2d_angles_np, effective_width):
    """
    Horizontally flips the image and adjusts landmarks and gaze yaw accordingly.
    """
    flipped_image = TF.hflip(image)
    
    # Flip landmarks
    flipped_landmarks_np = landmarks_np.copy()
    flipped_landmarks_np[:, 0] = effective_width - flipped_landmarks_np[:, 0]
    
    # Flip gaze yaw
    flipped_gaze_np = gaze_2d_angles_np.copy()
    flipped_gaze_np[1] *= -1 # Negate yaw (index 1)

    return flipped_image, flipped_landmarks_np, flipped_gaze_np

def normalize_landmarks(landmarks_np, effective_width, effective_height):
    """
    Normalizes landmarks to [0, 1] range and clips them.
    """
    normalized_landmarks = landmarks_np.copy()
    if effective_width > 0:
        normalized_landmarks[:, 0] /= effective_width
    else:
        normalized_landmarks[:, 0] = 0.0
    
    if effective_height > 0:
        normalized_landmarks[:, 1] /= effective_height
    else:
        normalized_landmarks[:, 1] = 0.0
        
    normalized_landmarks = np.clip(normalized_landmarks, 0.0, 1.0)
    return normalized_landmarks 

def crop_to_content(pil_image, landmarks_np, non_black_threshold=1):
    """
    Crops the PIL image to the bounding box of non-black pixels.
    """
    img_for_bbox_np = np.array(pil_image)
    
    if np.any(img_for_bbox_np > non_black_threshold):
        rows_with_content = np.any(img_for_bbox_np > non_black_threshold, axis=1)
        cols_with_content = np.any(img_for_bbox_np > non_black_threshold, axis=0)
        
        if np.any(rows_with_content) and np.any(cols_with_content):
            ymin_content, ymax_content = np.where(rows_with_content)[0][[0, -1]]
            xmin_content, xmax_content = np.where(cols_with_content)[0][[0, -1]]

            crop_x1 = xmin_content
            crop_y1 = ymin_content
            crop_x2 = xmax_content + 1
            crop_y2 = ymax_content + 1

            if crop_x2 > crop_x1 and crop_y2 > crop_y1:
                cropped_pil_image = pil_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                adjusted_landmarks_np = landmarks_np.copy()
                adjusted_landmarks_np[:, 0] -= crop_x1
                adjusted_landmarks_np[:, 1] -= crop_y1
                return cropped_pil_image, adjusted_landmarks_np
            else: print(f"Warning: Content crop box invalid for {getattr(pil_image, 'filename', 'image')}.")
        else: print(f"Warning: No valid row/col content for {getattr(pil_image, 'filename', 'image')}.")
    else: print(f"Warning: No non-black pixels for {getattr(pil_image, 'filename', 'image')}.")

    return pil_image, landmarks_np

def apply_clahe(pil_image, clip_limit=2.0, tile_grid_size=(8,8)):
    """
    Applies CLAHE to the PIL image.
    """
    img_np_for_clahe = np.array(pil_image)
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl_img_np = clahe_obj.apply(img_np_for_clahe)
    clahe_pil_image = Image.fromarray(cl_img_np)
    return clahe_pil_image 

def random_affine_with_landmarks(image, landmarks_np, gaze_2d_angles_np, degrees=(-20, 20), translate_fractions=(0.1, 0.1), scale_range=(0.8, 1.2)):
    """
    Applies random affine transformation (rotation, translation, scale) to the image 
    and adjusts landmarks and gaze yaw accordingly.
    """
    original_width, original_height = image.size
    center_x, center_y = original_width / 2.0, original_height / 2.0

    # Randomly sample transformation parameters
    angle = random.uniform(degrees[0], degrees[1])
    
    max_dx = translate_fractions[0] * original_width
    max_dy = translate_fractions[1] * original_height
    translate_x = random.uniform(-max_dx, max_dx)
    translate_y = random.uniform(-max_dy, max_dy)
    
    scale_factor = random.uniform(scale_range[0], scale_range[1])

    shear_params = [0.0, 0.0] # Default no shear

    # Apply affine transformation to the image
    transformed_image = TF.affine(image, 
                                  angle=angle, 
                                  translate=(translate_x, translate_y), 
                                  scale=scale_factor, 
                                  shear=shear_params)

    # Transform landmarks
    adjusted_landmarks = landmarks_np.copy().astype(np.float32)

    # Translate to origin (center of image) for scale/rotation/shear
    adjusted_landmarks[:, 0] -= center_x
    adjusted_landmarks[:, 1] -= center_y

    # Scale
    adjusted_landmarks *= scale_factor

    # Rotate
    angle_rad = math.radians(angle) # Use positive angle here as we are rotating points forward
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    x_rot = adjusted_landmarks[:, 0] * cos_a - adjusted_landmarks[:, 1] * sin_a
    y_rot = adjusted_landmarks[:, 0] * sin_a + adjusted_landmarks[:, 1] * cos_a
    adjusted_landmarks[:, 0] = x_rot
    adjusted_landmarks[:, 1] = y_rot

    # Translate back from origin and apply final translation
    adjusted_landmarks[:, 0] += center_x + translate_x
    adjusted_landmarks[:, 1] += center_y + translate_y
    
    # Adjust gaze by applying the 3D rotation corresponding to the 2D image rotation
    adjusted_gaze_np = gaze_2d_angles_np.copy()
    pitch, yaw = adjusted_gaze_np[0], adjusted_gaze_np[1]
    
    # Convert original (pitch, yaw) to a 3D vector.
    x = -math.cos(pitch) * math.sin(yaw)
    y = -math.sin(pitch)
    z = -math.cos(pitch) * math.cos(yaw)
    
    x_new = x * cos_a - y * sin_a
    y_new = x * sin_a + y * cos_a
    z_new = z
    
    # Convert the new 3D vector back to (pitch, yaw) format.
    new_pitch = math.asin(np.clip(-y_new, -1.0, 1.0))
    new_yaw = math.atan2(-x_new, -z_new)    
    
    adjusted_gaze_np[0] = new_pitch
    adjusted_gaze_np[1] = new_yaw
    
    # Clip landmarks to be within the image dimensions
    adjusted_landmarks[:, 0] = np.clip(adjusted_landmarks[:, 0], 0, original_width - 1)
    adjusted_landmarks[:, 1] = np.clip(adjusted_landmarks[:, 1], 0, original_height - 1)

    return transformed_image, adjusted_landmarks, adjusted_gaze_np

def landmarks_smoothing(landmarks_np, smoothing_factor=0.01, landmark_bounds=(0.0, 1.0)):
    """
    Applies label smoothing to normalized facial landmarks by adding small random noise.
    """
    if smoothing_factor <= 0:
        return landmarks_np
    
    # Generate Gaussian noise with the same shape as landmarks
    noise = np.random.normal(0, smoothing_factor, landmarks_np.shape).astype(np.float32)
    
    # Add noise to landmarks
    smoothed_landmarks = landmarks_np + noise
    
    # Clip to ensure landmarks stay within valid bounds
    smoothed_landmarks = np.clip(smoothed_landmarks, landmark_bounds[0], landmark_bounds[1])
    
    return smoothed_landmarks
