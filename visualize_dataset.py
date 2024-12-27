import os
import glob
import h5py
import numpy as np
import open3d as o3d
import random

def load_modelnet_data(partition, cat=40):
    BASE_DIR = '/home/virgantara/PythonProjects/DualGraphPoint'
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data, new_all_data = [], []
    all_label, new_all_label = [], []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', f'ply_data_{partition}*.h5')):
        f = h5py.File(h5_name, 'r')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    if cat == 10:
        for i in range(len(all_label)):
            if all_label[i] in [1, 2, 8, 12, 14, 22, 23, 30, 33, 35]:
                # Selected categories for ModelNet10
                new_all_data.append(all_data[i])
                new_all_label.append(all_label[i])
        all_data = np.array(new_all_data)
        all_label = np.array(new_all_label)
    return all_data, all_label

def visualize_point_cloud_open3d(point_cloud, class_label, class_names):
    """Visualizes a single point cloud using Open3D."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    # Optional: Add color for visualization (use Z-axis for colormap)
    # colors = (point_cloud[:, 2] - np.min(point_cloud[:, 2])) / (np.max(point_cloud[:, 2]) - np.min(point_cloud[:, 2]))
    # pcd.colors = o3d.utility.Vector3dVector(np.stack([colors, colors, colors], axis=1))
    print(f"Visualizing Class: {class_names[class_label]}")
    o3d.visualization.draw_geometries([pcd])

def load_data_partseg(partition):
    BASE_DIR = '/home/virgantara/PythonProjects/DualGraphPoint'
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    all_seg = []
    if partition == 'trainval':
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*train*.h5')) \
               + glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', '*val*.h5'))
    else:
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', f'*{partition}*.h5'))
    for h5_name in file:
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        seg = f['pid'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
        all_seg.append(seg)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_seg = np.concatenate(all_seg, axis=0)
    return all_data, all_label, all_seg


def visualize_point_cloud_with_segmentation(point_cloud, segmentation, num_segments):
    """Visualize a point cloud with segmentation labels using Open3D."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    
    # Generate a color map for segmentation labels
    color_map = np.random.rand(num_segments, 3)  # Random colors for each segment
    colors = color_map[segmentation]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.visualization.draw_geometries([pcd])

def vis_modelnet():
    # Load the dataset
    partition = 'train'  # or 'test'
    class_names = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
                   'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar',
                   'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano',
                   'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent',
                   'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
    all_data, all_label = load_modelnet_data(partition)
    airplane_label = class_names.index('table')  # Get the label index for "airplane"
    airplane_indices = np.where(all_label == airplane_label)[0]

    if len(airplane_indices) > 0:
        # Select a random "airplane" sample
        idx = np.random.choice(airplane_indices)
        point_cloud = all_data[idx]
        label = all_label[idx][0]

        # Visualize the sample
        visualize_point_cloud_open3d(point_cloud, label, class_names)
    else:
        print("No samples found for the 'airplane' class.")

def vis_shapenet():
    # Load ShapeNetPart data
    partition = 'test'  # Choose 'train', 'val', 'test', or 'trainval'
    all_data, all_label, all_seg = load_data_partseg(partition)

    # Randomly select a sample
    idx = random.randint(0, len(all_data) - 1)
    point_cloud = all_data[idx]
    segmentation = all_seg[idx]
    label = all_label[idx][0]  # ShapeNetPart class label

    # Number of segments depends on the dataset (usually up to 50 for ShapeNetPart)
    num_segments = segmentation.max() + 1
    print(f"Visualizing ShapeNetPart sample: Class {label}, Segments: {num_segments}")

    # Visualize the selected point cloud with segmentation
    visualize_point_cloud_with_segmentation(point_cloud, segmentation, num_segments)
# Main program
if __name__ == "__main__":
    
    vis_shapenet()