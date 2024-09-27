import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Sample code to visualize point clouds and highlight specific regions


def load_point_cloud_from_array(points, labels):
    # Load point cloud from numpy arrays
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # Map labels to colors
    unique_labels = np.unique(labels)
    colors = plt.get_cmap("tab20")(labels / float(max(unique_labels)))[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def highlight_improvements(gt_labels, pred_labels, points):
    # Create a point cloud with improvements highlighted
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Initialize all points to gray (no improvement)
    colors = np.ones((points.shape[0], 3)) * [0.5, 0.5, 0.5]  # Gray for non-highlighted points

    # Highlight improved or mismatched regions
    improvement_indices = np.where(gt_labels != pred_labels)[0]  # Find mismatched points
    colors[improvement_indices] = [1, 0, 0]  # Red for mismatches (or improvements)
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def visualize_point_cloud(pcd):
    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

# Example usage:
# Load your point cloud file (replace with the actual file path of your point cloud)
output_path = "/home/virgantara/PythonProjects/DynamicCrossNet/outputs"
model_path = "/exp_partseg_adacrossnet/"
file_path = output_path+model_path+"visualization/airplane/airplane_5_pred_0.9322.ply"

# Load your point cloud data (replace these with your actual data)
points = np.random.rand(2048, 3)  # Replace with your point cloud data
gt_labels = np.random.randint(0, 10, size=(2048,))  # Replace with your ground truth segmentation labels
pred_labels = np.random.randint(0, 10, size=(2048,))  # Replace with your predicted segmentation labels
# print(gt_labels.shape)
# Highlight the differences between ground truth and prediction
highlighted_pcd = highlight_improvements(gt_labels, pred_labels, points)

# Visualize the point cloud with improvements/mismatches highlighted
visualize_point_cloud(highlighted_pcd)


# pcd = load_point_cloud(file_path)

# # Highlight specific regions (replace with actual indices you want to highlight)
# highlight_indices = np.random.choice(len(pcd.points), size=500, replace=False)  # Replace with actual logic
# highlight_regions(pcd, highlight_indices)

# # Visualize the point cloud
# visualize_point_cloud(pcd)
