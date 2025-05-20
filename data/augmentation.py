import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import random


def crop_image(image, original_landmarks_np, is_train, train_crop_padding_range, test_crop_padding_factor, train_random_shift_fraction):
    """
    Crops the image based on facial landmarks.
    Applies padding and optional random shifting for training.
    """
    original_width, original_height = image.size
    current_image = image
    current_landmarks_np = original_landmarks_np.copy()
    effective_width, effective_height = original_width, original_height

    padding_factor_to_use = 0.0
    apply_random_shift = False

    if is_train:
        padding_factor_to_use = random.uniform(train_crop_padding_range[0], train_crop_padding_range[1])
        if train_random_shift_fraction > 0.0:
            apply_random_shift = True
    else:  # is_test
        padding_factor_to_use = test_crop_padding_factor

    x_min_lm, y_min_lm = np.min(original_landmarks_np, axis=0)
    x_max_lm, y_max_lm = np.max(original_landmarks_np, axis=0)

    bbox_width = x_max_lm - x_min_lm
    bbox_height = y_max_lm - y_min_lm
    
    bbox_width = max(0, bbox_width)
    bbox_height = max(0, bbox_height)

    pad_w_half = (bbox_width * padding_factor_to_use) / 2.0
    pad_h_half = (bbox_height * padding_factor_to_use) / 2.0

    f_padded_crop_x1 = x_min_lm - pad_w_half
    f_padded_crop_y1 = y_min_lm - pad_h_half
    f_padded_crop_x2 = x_max_lm + pad_w_half
    f_padded_crop_y2 = y_max_lm + pad_h_half
    
    final_f_crop_x1, final_f_crop_y1 = f_padded_crop_x1, f_padded_crop_y1
    final_f_crop_x2, final_f_crop_y2 = f_padded_crop_x2, f_padded_crop_y2

    if apply_random_shift:
        dx_min_lm_constraint = -pad_w_half 
        dx_max_lm_constraint = pad_w_half
        dy_min_lm_constraint = -pad_h_half
        dy_max_lm_constraint = pad_h_half
        
        dx_min_img_constraint = -f_padded_crop_x1
        dx_max_img_constraint = original_width - f_padded_crop_x2
        dy_min_img_constraint = -f_padded_crop_y1
        dy_max_img_constraint = original_height - f_padded_crop_y2

        actual_dx_min = max(dx_min_lm_constraint, dx_min_img_constraint)
        actual_dx_max = min(dx_max_lm_constraint, dx_max_img_constraint)
        
        actual_dy_min = max(dy_min_lm_constraint, dy_min_img_constraint)
        actual_dy_max = min(dy_max_lm_constraint, dy_max_img_constraint)

        random_dx = 0.0
        random_dy = 0.0

        if actual_dx_max > actual_dx_min:
            padded_box_width = f_padded_crop_x2 - f_padded_crop_x1
            if padded_box_width > 1e-6 : 
                max_abs_shift_x = train_random_shift_fraction * padded_box_width
                shift_limit_x_min = max(actual_dx_min, -max_abs_shift_x)
                shift_limit_x_max = min(actual_dx_max, max_abs_shift_x)
                if shift_limit_x_max > shift_limit_x_min:
                     random_dx = random.uniform(shift_limit_x_min, shift_limit_x_max)
        
        if actual_dy_max > actual_dy_min:
            padded_box_height = f_padded_crop_y2 - f_padded_crop_y1
            if padded_box_height > 1e-6:
                max_abs_shift_y = train_random_shift_fraction * padded_box_height
                shift_limit_y_min = max(actual_dy_min, -max_abs_shift_y)
                shift_limit_y_max = min(actual_dy_max, max_abs_shift_y)
                if shift_limit_y_max > shift_limit_y_min:
                    random_dy = random.uniform(shift_limit_y_min, shift_limit_y_max)
        
        final_f_crop_x1 += random_dx
        final_f_crop_y1 += random_dy
        final_f_crop_x2 += random_dx
        final_f_crop_y2 += random_dy
    
    crop_x1_final = int(round(np.maximum(0, final_f_crop_x1)))
    crop_y1_final = int(round(np.maximum(0, final_f_crop_y1)))
    crop_x2_final = int(round(np.minimum(original_width, final_f_crop_x2)))
    crop_y2_final = int(round(np.minimum(original_height, final_f_crop_y2)))

    if crop_x1_final < crop_x2_final and crop_y1_final < crop_y2_final:
        cropped_image_candidate = image.crop((crop_x1_final, crop_y1_final, crop_x2_final, crop_y2_final))
        if cropped_image_candidate.width > 0 and cropped_image_candidate.height > 0:
            current_image = cropped_image_candidate
            effective_width, effective_height = current_image.size
            current_landmarks_np[:, 0] -= crop_x1_final
            current_landmarks_np[:, 1] -= crop_y1_final
        else:
            print(f"Warning: Crop resulted in zero dimension. Original image used. Box:({crop_x1_final},{crop_y1_final},{crop_x2_final},{crop_y2_final})")
    else:
        print(f"Warning: Invalid crop box calculated. Original image used. Box:({crop_x1_final},{crop_y1_final},{crop_x2_final},{crop_y2_final})")
    
    return current_image, current_landmarks_np, effective_width, effective_height


def horizontal_flip(image, landmarks_np, effective_width):
    """
    Horizontally flips the image and adjusts landmarks accordingly.
    """
    flipped_image = TF.hflip(image)
    flipped_landmarks_np = landmarks_np.copy()
    flipped_landmarks_np[:, 0] = effective_width - flipped_landmarks_np[:, 0]
    return flipped_image, flipped_landmarks_np

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