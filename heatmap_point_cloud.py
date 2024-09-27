import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def load_point_cloud(points, labels):
    # Create a point cloud from numpy arrays
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Map labels to colors
    unique_labels = np.unique(labels)
    colors = plt.get_cmap("tab20")(labels / float(max(unique_labels)))[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def compute_error_heatmap(gt_labels, pred_labels):
    # Compute an error mask (1 for incorrect, 0 for correct)
    error = np.abs(gt_labels - pred_labels)
    return error

def apply_heatmap_to_point_cloud(pcd, errors):
    # Normalize errors to a 0-1 range for color mapping
    normalized_errors = (errors - np.min(errors)) / (np.max(errors) - np.min(errors))
    
    # Apply heatmap color based on the error value (blue for low error, red for high error)
    colors = plt.get_cmap("coolwarm")(normalized_errors)[:, :3]
    
    # Assign colors to the point cloud
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
# Assuming you have the point cloud coordinates (points), ground truth labels, and predicted labels
# points: N x 3 numpy array of point cloud coordinates
# gt_labels: N-length numpy array of ground truth segmentation labels
# pred_labels: N-length numpy array of predicted segmentation labels

base_path = '/home/virgantara/PythonProjects/DynamicCrossNet/outputs'

model_path = '/exp_partseg_pointnet2/'

pc = np.loadtxt(base_path+model_path+'/visualization/earphone/earphone_5_pred_0.5708.txt')
points = pc[:,:3].astype(float)
pred_labels = pc[:,-1].astype(int)

pc_gt = np.loadtxt(base_path+model_path+'/visualization/earphone/earphone_5_gt.txt')
gt_labels = pc_gt[:,-1].astype(int)

# Load your point cloud data (replace these with your actual data)
# points = np.random.rand(2048, 3)  # Replace with your point cloud data
# gt_labels = np.random.randint(0, 10, size=(2048,))  # Replace with ground truth labels
# pred_labels = np.random.randint(0, 10, size=(2048,))  # Replace with predicted labels

# Step 1: Compute the error heatmap
errors = compute_error_heatmap(gt_labels, pred_labels)

# Step 2: Load the point cloud with ground truth data
pcd = load_point_cloud(points, gt_labels)

# Step 3: Apply the error heatmap to the point cloud
pcd_with_heatmap = apply_heatmap_to_point_cloud(pcd, errors)

# Step 4: Visualize the point cloud with heatmap
visualize_point_cloud(pcd_with_heatmap)
