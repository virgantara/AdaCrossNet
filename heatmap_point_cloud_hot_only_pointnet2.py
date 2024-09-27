import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def load_point_cloud(points, labels):
    # Create a point cloud from numpy arrays
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Map labels to colors (grayscale for now, since we will highlight significant differences)
    colors = np.ones((len(labels), 3)) * 0.7  # Gray for non-highlighted points
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def compute_difference_heatmap(labels_A, labels_B):
    # Compute absolute difference between method A and method B
    differences = np.abs(labels_A - labels_B)
    return differences

def apply_high_difference_highlighting(pcd, differences, threshold):
    # Identify high-difference points (above the threshold)
    high_diff_indices = np.where(differences >= threshold)[0]
    
    # Set the color for high-difference points (e.g., red)
    colors = np.asarray(pcd.colors)
    colors[high_diff_indices] = [1, 0, 0]  # Red for high differences
    
    # Apply the new colors to the point cloud
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def visualize_point_cloud(pcd):
    # Create a visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add the point cloud to the visualizer
    vis.add_geometry(pcd)

    # Set the point size to be larger
    render_option = vis.get_render_option()
    render_option.point_size = 11.0  # Increase point size (default is 1.0)

    # Run the visualizer
    vis.run()
    vis.destroy_window()

# Example usage:
# Assuming you have the point cloud coordinates (points), method A and method B predicted labels
# points: N x 3 numpy array of point cloud coordinates
# labels_A: N-length numpy array of segmentation labels from method A
# labels_B: N-length numpy array of segmentation labels from method B


base_path = '/home/virgantara/PythonProjects/DynamicCrossNet/outputs'

model_path = '/exp_partseg_pointnet2/'

pc = np.loadtxt(base_path+model_path+'/visualization/airplane/airplane_5_pred_0.9254.txt')
points = pc[:,:3].astype(float)
labels_A = pc[:,-1].astype(int)

pc_gt = np.loadtxt(base_path+model_path+'/visualization/airplane/airplane_5_gt.txt')
labels_B = pc_gt[:,-1].astype(int)


# Load your point cloud data (replace these with your actual data)
# points = np.random.rand(2048, 3)  # Replace with your point cloud data
# labels_A = np.random.randint(0, 10, size=(2048,))  # Replace with Method A predicted labels
# labels_B = np.random.randint(0, 10, size=(2048,))  # Replace with Method B predicted labels

# Step 1: Compute the differences between Method A and Method B
differences = compute_difference_heatmap(labels_A, labels_B)

# Step 2: Load the point cloud
pcd = load_point_cloud(points, labels_A)

# Step 3: Highlight high differences (threshold can be tuned)
threshold = 2  # Adjust this threshold as needed (e.g., 1 or higher for significant differences)
pcd_with_highlighting = apply_high_difference_highlighting(pcd, differences, threshold)

# Step 4: Visualize the point cloud with highlighted high differences
visualize_point_cloud(pcd_with_highlighting)
