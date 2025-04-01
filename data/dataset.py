from torch.utils.data import Dataset
import os
import numpy as np
import torch
from PIL import Image

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


class AutoPOSE(BaseDataset):
    """
    Dataset for AutoPOSE.
    This dataset loads pose data from a directory structure.
    
    Args:
        data_dir (str): Directory containing the dataset.
        annotations_file (str): Path to the annotations file.
        transform (callable, optional): Optional transform to be applied on images.
        target_transform (callable, optional): Optional transform to be applied on targets.
    """
    def __init__(self, data_dir, annotations_file, transform=None, target_transform=None):
        super().__init__(transform)
        self.data_dir = data_dir
        self.target_transform = target_transform
        
        # Load annotations
        self.annotations = self._load_annotations(annotations_file)
        
        # Get list of image files and corresponding pose data
        self.image_paths, self.pose_data = self._parse_annotations()
        
    def _load_annotations(self, annotations_file):
        # Load annotations from file (can be CSV, JSON, etc.)
        # For example, using a CSV file similar to the face landmarks example:
        try:
            import pandas as pd
            return pd.read_csv(annotations_file)
        except Exception as e:
            print(f"Error loading annotations: {e}")
            # Fallback to an empty DataFrame if file can't be loaded
            import pandas as pd
            return pd.DataFrame()
            
    def _parse_annotations(self):
        # Extract image paths and pose data from annotations
        image_paths = []
        pose_data = []
        
        # Assuming annotations format similar to face landmarks tutorial
        # with image_name in first column and pose keypoints in subsequent columns
        for idx, row in self.annotations.iterrows():
            img_name = row[0]  # First column contains image name
            img_path = os.path.join(self.data_dir, img_name)
            
            # Extract pose keypoints from remaining columns and convert to numpy array
            keypoints = np.array(row[1:], dtype=np.float32)
            keypoints = keypoints.reshape(-1, 2)  # Reshape to (num_keypoints, 2)
            
            image_paths.append(img_path)
            pose_data.append(keypoints)
            
        return image_paths, pose_data
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        # Load image
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert('RGB')
        
        # Get pose data
        pose = self.pose_data[index]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        if self.target_transform:
            pose = self.target_transform(pose)
        else:
            # Convert to tensor by default
            pose = torch.from_numpy(pose)
            
        # Return sample as a dictionary
        return {
            'image': image,
            'pose': pose,
            'path': img_path
        }
        