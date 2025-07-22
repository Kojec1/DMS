import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from tqdm import tqdm
import argparse

# Add parent directory to path to import dataset classes
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_gaze_data_from_mat(mat_file_path):
    """Load gaze data from a single .mat file."""
    gaze_angles = []
    
    try:
        with h5py.File(mat_file_path, 'r') as f:
            if 'Data' not in f or 'label' not in f['Data']:
                print(f"Warning: Unexpected structure in {mat_file_path}")
                return np.array([])
            
            labels = f['Data']['label']
            n_samples = labels.shape[0]
            
            print(f"Loading {n_samples} samples from {os.path.basename(mat_file_path)}")
            
            for i in range(n_samples):
                label = labels[i]
                # First two elements are gaze angles [pitch, yaw] in radians
                pitch, yaw = label[0], label[1]
                gaze_angles.append([pitch, yaw])
                
    except Exception as e:
        print(f"Error reading {mat_file_path}: {e}")
        return np.array([])
    
    return np.array(gaze_angles)

def analyze_gaze_distribution(data_dir):
    """Analyze gaze distribution across all available MPII .mat files."""
    # Find all .mat files
    mat_files = glob(os.path.join(data_dir, "p*.mat"))
    
    if not mat_files:
        print(f"No .mat files found in {data_dir}")
        print("Please ensure MPII dataset .mat files are available")
        return
    
    print(f"Found {len(mat_files)} .mat files:")
    for f in mat_files:
        print(f"  - {os.path.basename(f)}")
    
    # Collect all gaze data
    all_gaze_angles = []
    
    for mat_file in tqdm(mat_files, desc="Loading gaze data"):
        gaze_data = load_gaze_data_from_mat(mat_file)
        if len(gaze_data) > 0:
            all_gaze_angles.append(gaze_data)
    
    if not all_gaze_angles:
        print("No gaze data found!")
        return
    
    # Combine all data
    all_gaze_angles = np.vstack(all_gaze_angles)
    print(f"\nTotal samples: {len(all_gaze_angles)}")
    
    # Convert from radians to degrees
    gaze_degrees = np.degrees(all_gaze_angles)
    
    pitch_deg = gaze_degrees[:, 0]
    yaw_deg = gaze_degrees[:, 1]
    
    # Display statistics
    print("\n" + "="*60)
    print("GAZE ANGLE DISTRIBUTION ANALYSIS")
    print("="*60)
    
    print(f"\nPITCH STATISTICS (degrees):")
    print(f"  Min:    {pitch_deg.min():.2f}°")
    print(f"  Max:    {pitch_deg.max():.2f}°")
    print(f"  Mean:   {pitch_deg.mean():.2f}°")
    print(f"  Std:    {pitch_deg.std():.2f}°")
    print(f"  Median: {np.median(pitch_deg):.2f}°")
    
    print(f"\nYAW STATISTICS (degrees):")
    print(f"  Min:    {yaw_deg.min():.2f}°")
    print(f"  Max:    {yaw_deg.max():.2f}°")
    print(f"  Mean:   {yaw_deg.mean():.2f}°")
    print(f"  Std:    {yaw_deg.std():.2f}°")
    print(f"  Median: {np.median(yaw_deg):.2f}°")
    
    # Create visualizations
    create_gaze_visualizations(pitch_deg, yaw_deg)
    
    return pitch_deg, yaw_deg

def create_gaze_visualizations(pitch_deg, yaw_deg):
    """
    Create heatmap and distribution visualizations for gaze data.
    
    Args:
        pitch_deg: Pitch angles in degrees
        yaw_deg: Yaw angles in degrees
    """
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("viridis")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 2D Heatmap of gaze distribution
    ax1 = plt.subplot(2, 3, 1)
    plt.hist2d(yaw_deg, pitch_deg, bins=50, cmap='viridis', density=True)
    plt.colorbar(label='Density')
    plt.xlabel('Yaw (degrees)')
    plt.ylabel('Pitch (degrees)')
    plt.title('2D Gaze Distribution Heatmap')
    plt.grid(True, alpha=0.3)
    
    # 2. Pitch distribution histogram
    ax2 = plt.subplot(2, 3, 2)
    plt.hist(pitch_deg, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Pitch (degrees)')
    plt.ylabel('Frequency')
    plt.title('Pitch Angle Distribution')
    plt.grid(True, alpha=0.3)
    plt.axvline(pitch_deg.mean(), color='red', linestyle='--', 
                label=f'Mean: {pitch_deg.mean():.2f}°')
    plt.legend()
    
    # 3. Yaw distribution histogram
    ax3 = plt.subplot(2, 3, 3)
    plt.hist(yaw_deg, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.xlabel('Yaw (degrees)')
    plt.ylabel('Frequency')
    plt.title('Yaw Angle Distribution')
    plt.grid(True, alpha=0.3)
    plt.axvline(yaw_deg.mean(), color='red', linestyle='--', 
                label=f'Mean: {yaw_deg.mean():.2f}°')
    plt.legend()
    
    # 4. Joint distribution with kde
    ax4 = plt.subplot(2, 3, 4)
    sns.kdeplot(x=yaw_deg, y=pitch_deg, fill=True, cmap='viridis', levels=20)
    plt.xlabel('Yaw (degrees)')
    plt.ylabel('Pitch (degrees)')
    plt.title('Gaze Distribution (KDE)')
    plt.grid(True, alpha=0.3)
    
    # 5. Box plots
    ax5 = plt.subplot(2, 3, 5)
    box_data = [pitch_deg, yaw_deg]
    box_labels = ['Pitch', 'Yaw']
    plt.boxplot(box_data, labels=box_labels)
    plt.ylabel('Angle (degrees)')
    plt.title('Gaze Angle Box Plots')
    plt.grid(True, alpha=0.3)
    
    # 6. Scatter plot with marginals
    ax6 = plt.subplot(2, 3, 6)
    # Sample data for better visualization if we have too many points
    if len(pitch_deg) > 10000:
        indices = np.random.choice(len(pitch_deg), 10000, replace=False)
        pitch_sample = pitch_deg[indices]
        yaw_sample = yaw_deg[indices]
    else:
        pitch_sample = pitch_deg
        yaw_sample = yaw_deg
    
    plt.scatter(yaw_sample, pitch_sample, alpha=0.3, s=1, c='blue')
    plt.xlabel('Yaw (degrees)')
    plt.ylabel('Pitch (degrees)')
    plt.title('Gaze Scatter Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = 'gaze_distribution_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as: {output_path}")
    
    # Show the plot
    plt.show()
    
    # Create additional detailed heatmap
    create_detailed_heatmap(pitch_deg, yaw_deg)

def create_detailed_heatmap(pitch_deg, yaw_deg):
    """
    Create a more detailed heatmap with custom binning.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # High-resolution heatmap
    H, xedges, yedges = np.histogram2d(yaw_deg, pitch_deg, bins=100)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    im1 = ax1.imshow(H.T, extent=extent, origin='lower', cmap='hot', aspect='auto')
    ax1.set_xlabel('Yaw (degrees)')
    ax1.set_ylabel('Pitch (degrees)')
    ax1.set_title('High-Resolution Gaze Heatmap')
    plt.colorbar(im1, ax=ax1, label='Count')
    
    # Normalized heatmap
    H_norm = H / H.sum()
    im2 = ax2.imshow(H_norm.T, extent=extent, origin='lower', cmap='plasma', aspect='auto')
    ax2.set_xlabel('Yaw (degrees)')
    ax2.set_ylabel('Pitch (degrees)')
    ax2.set_title('Normalized Gaze Probability Density')
    plt.colorbar(im2, ax=ax2, label='Probability Density')
    
    plt.tight_layout()
    plt.savefig('detailed_gaze_heatmap.png', dpi=300, bbox_inches='tight')
    print("Detailed heatmap saved as: detailed_gaze_heatmap.png")
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description="MPII Gaze Dataset Distribution Analyzer")
    parser.add_argument("--data_dir", type=str, default="../notebooks", help="Path to the MPII dataset directory")
    return parser.parse_args()

def main():
    """Main function to run the gaze distribution analysis."""
    args = parse_args()
    data_dir = args.data_dir
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found.")
        return
    
    # Run analysis
    pitch_deg, yaw_deg = analyze_gaze_distribution(data_dir)
    
    if pitch_deg is not None and yaw_deg is not None:
        print("\nAnalysis completed successfully!")
        print(f"Generated visualizations:")
        print(f"  - gaze_distribution_analysis.png")
        print(f"  - detailed_gaze_heatmap.png")

if __name__ == "__main__":
    main()
