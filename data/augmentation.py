import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import random
import cv2
import math


def horizontal_flip(image, landmarks_np, gaze_2d_angles_np=None, head_2d_angles_np=None, effective_width=None):
    """
    Horizontally flips the image and adjusts landmarks, gaze, and head pose accordingly.
    Supports both 2D gaze/head angles and 3D gaze direction vectors.
    """
    flipped_image = TF.hflip(image)
    
    # Flip landmarks
    flipped_landmarks_np = landmarks_np.copy()
    if effective_width is not None:
        flipped_landmarks_np[:, 0] = effective_width - flipped_landmarks_np[:, 0]
    
    # Flip 2D gaze yaw if provided
    flipped_gaze_2d_np = None
    if gaze_2d_angles_np is not None:
        flipped_gaze_2d_np = gaze_2d_angles_np.copy()
        flipped_gaze_2d_np[1] *= -1  # Negate yaw (index 1)
    
    # Flip 2D head pose yaw if provided
    flipped_head_2d_np = None
    if head_2d_angles_np is not None:
        flipped_head_2d_np = head_2d_angles_np.copy()
        flipped_head_2d_np[1] *= -1  # Negate yaw (index 1)

    return flipped_image, flipped_landmarks_np, flipped_gaze_2d_np, flipped_head_2d_np

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

def crop_to_landmarks(pil_image, landmarks_np, padding_ratio=0.3, translation_ratio=0.2):
    """
    Crops the PIL image to the bounding box of facial landmarks with padding.
    """
    if landmarks_np.size == 0:
        print("Warning: No landmarks provided for cropping.")
        return pil_image, landmarks_np
    
    # Get image dimensions
    img_width, img_height = pil_image.size
    
    # Calculate bounding box of landmarks
    x_min = np.min(landmarks_np[:, 0])
    x_max = np.max(landmarks_np[:, 0])
    y_min = np.min(landmarks_np[:, 1])
    y_max = np.max(landmarks_np[:, 1])
    
    # Calculate current bounding box dimensions
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    
    # Add padding
    padding_x = bbox_width * padding_ratio
    padding_y = bbox_height * padding_ratio
    
    # Calculate base crop coordinates with padding
    base_crop_x1 = x_min - padding_x
    base_crop_y1 = y_min - padding_y
    base_crop_x2 = x_max + padding_x
    base_crop_y2 = y_max + padding_y
    
    # Calculate crop dimensions
    crop_width = base_crop_x2 - base_crop_x1
    crop_height = base_crop_y2 - base_crop_y1
    
    # Add random translation if specified
    if translation_ratio > 0:
        # Calculate maximum translation based on face size
        max_translation_x = bbox_width * translation_ratio
        max_translation_y = bbox_height * translation_ratio
        
        # Generate random translation offsets
        translation_x = random.uniform(-max_translation_x, max_translation_x)
        translation_y = random.uniform(-max_translation_y, max_translation_y)
        
        # Apply translation
        base_crop_x1 += translation_x
        base_crop_y1 += translation_y
        base_crop_x2 += translation_x
        base_crop_y2 += translation_y
    
    # Ensure crop box stays within image boundaries
    crop_x1 = max(0, int(base_crop_x1))
    crop_y1 = max(0, int(base_crop_y1))
    crop_x2 = min(img_width, int(base_crop_x2))
    crop_y2 = min(img_height, int(base_crop_y2))
    
    # Crop the image
    cropped_pil_image = pil_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    
    # Adjust landmarks relative to the new crop region
    adjusted_landmarks_np = landmarks_np.copy()
    adjusted_landmarks_np[:, 0] -= crop_x1
    adjusted_landmarks_np[:, 1] -= crop_y1
    
    return cropped_pil_image, adjusted_landmarks_np

def apply_clahe(pil_image, clip_limit=2.0, tile_grid_size=(8,8)):
    """
    Applies CLAHE to the PIL image. Handles both grayscale and RGB images.
    """
    if pil_image.mode == 'L':
        # Grayscale image
        img_np_for_clahe = np.array(pil_image)
        clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl_img_np = clahe_obj.apply(img_np_for_clahe)
        clahe_pil_image = Image.fromarray(cl_img_np)
    elif pil_image.mode == 'RGB':
        # RGB image
        img_np_rgb = np.array(pil_image)
        # Convert RGB to LAB color space
        img_np_lab = cv2.cvtColor(img_np_rgb, cv2.COLOR_RGB2LAB)
        # Split LAB channels
        l_channel, a_channel, b_channel = cv2.split(img_np_lab)
        # Apply CLAHE to the L-channel
        clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl_l_channel = clahe_obj.apply(l_channel)
        # Merge the CLAHE-enhanced L-channel back with A and B channels
        merged_lab_channels = cv2.merge((cl_l_channel, a_channel, b_channel))
        # Convert back to RGB color space
        final_rgb_img_np = cv2.cvtColor(merged_lab_channels, cv2.COLOR_LAB2RGB)
        clahe_pil_image = Image.fromarray(final_rgb_img_np)
    else:
        # For other modes, return the original image
        clahe_pil_image = pil_image

    return clahe_pil_image

def random_affine_with_landmarks(image, landmarks_np, gaze_2d_angles_np=None, head_2d_angles_np=None, degrees=(-20, 20), translate_fractions=(0.1, 0.1), scale_range=(0.8, 1.2)):
    """
    Applies random affine transformation (rotation, translation, scale) to the image 
    and adjusts landmarks, gaze, and head pose accordingly.
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

    # Apply affine transformation to the image
    transformed_image = TF.affine(image, 
                                  angle=angle, 
                                  translate=(translate_x, translate_y), 
                                  scale=scale_factor,
                                  shear=(0, 0))

    # Transform landmarks
    adjusted_landmarks = landmarks_np.copy().astype(np.float32)

    # Translate to origin (center of image) for scale/rotation
    adjusted_landmarks[:, 0] -= center_x
    adjusted_landmarks[:, 1] -= center_y

    # Scale
    adjusted_landmarks *= scale_factor

    # Rotate
    angle_rad = math.radians(angle)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    x_rot = adjusted_landmarks[:, 0] * cos_a - adjusted_landmarks[:, 1] * sin_a
    y_rot = adjusted_landmarks[:, 0] * sin_a + adjusted_landmarks[:, 1] * cos_a
    adjusted_landmarks[:, 0] = x_rot
    adjusted_landmarks[:, 1] = y_rot

    # Translate back from origin and apply final translation
    adjusted_landmarks[:, 0] += center_x + translate_x
    adjusted_landmarks[:, 1] += center_y + translate_y

    # Clip landmarks to be within the image dimensions
    adjusted_landmarks[:, 0] = np.clip(adjusted_landmarks[:, 0], 0, original_width - 1)
    adjusted_landmarks[:, 1] = np.clip(adjusted_landmarks[:, 1], 0, original_height - 1)
    
    # Adjust 2D gaze if provided
    adjusted_gaze_2d_np = None
    if gaze_2d_angles_np is not None:
        adjusted_gaze_2d_np = gaze_2d_angles_np.copy()
        pitch, yaw = adjusted_gaze_2d_np[0], adjusted_gaze_2d_np[1]
        
        # Convert original (pitch, yaw) to a 3D vector.
        x = -math.cos(pitch) * math.sin(yaw)
        y = -math.sin(pitch)
        z = -math.cos(pitch) * math.cos(yaw)
        
        # Counter-clockwise rotation
        x_new = x * cos_a + y * sin_a
        y_new = -x * sin_a + y * cos_a
        z_new = z
        
        # Convert the new 3D vector back to (pitch, yaw) format.
        new_pitch = math.asin(np.clip(-y_new, -1.0, 1.0))
        new_yaw = math.atan2(-x_new, -z_new)    
        
        adjusted_gaze_2d_np[0] = new_pitch
        adjusted_gaze_2d_np[1] = new_yaw

    # Adjust 2D head pose if provided (similar to gaze adjustment)
    adjusted_head_2d_np = None
    if head_2d_angles_np is not None:
        adjusted_head_2d_np = head_2d_angles_np.copy()
        pitch, yaw = adjusted_head_2d_np[0], adjusted_head_2d_np[1]
        
        # Convert head (pitch, yaw) to a 3D vector (forward direction)
        x = -math.cos(pitch) * math.sin(yaw)
        y = -math.sin(pitch)
        z = -math.cos(pitch) * math.cos(yaw)
        
        # Apply rotation (counter-clockwise)
        x_new = x * cos_a + y * sin_a
        y_new = -x * sin_a + y * cos_a
        z_new = z
        
        # Convert back to (pitch, yaw)
        new_pitch = math.asin(np.clip(-y_new, -1.0, 1.0))
        new_yaw = math.atan2(-x_new, -z_new)
        
        adjusted_head_2d_np[0] = new_pitch
        adjusted_head_2d_np[1] = new_yaw

    return transformed_image, adjusted_landmarks, adjusted_gaze_2d_np, adjusted_head_2d_np

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
