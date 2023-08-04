import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def remove_planes(pcd, distance_threshold=0.3):
    while True:
        _, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                                  ransac_n=3,
                                                  num_iterations=4000)
        outliers = pcd.select_by_index(inliers, invert=True)

        if len(inliers) < 0.5 * len(pcd.points):
            break

        pcd = outliers

    return pcd

def clustering(pcd, eps=0.3, min_points=10):  
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    max_label = labels.max()
    # All black
    colors = np.zeros((len(pcd.points), 3))
    bounding_boxes = []
    convex_hulls = []
    triangle_meshes = []

    for i in range(max_label + 1):
        valid_points = np.where(labels == i)[0]
        if len(valid_points) > 0:
            # Add colors
            color = plt.get_cmap("tab20")(i / (max_label + 1))[:3]
            colors[valid_points] = color
            cluster = pcd.select_by_index(valid_points)
            bbox = cluster.get_axis_aligned_bounding_box()
            bbox.color = color
            bounding_boxes.append(bbox)
            # Convex hull
            if len(valid_points) >= 3:
                hull, _ = cluster.compute_convex_hull()
                hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
                hull_ls.paint_uniform_color((0, 1, 0))
                convex_hulls.append(hull_ls)

                # Triangulation
                mesh = hull.simplify_quadric_decimation(1000)
                triangle_meshes.append(mesh)

    pcd = pcd.select_by_index(np.where(labels >= 0)[0])
    pcd.colors = o3d.utility.Vector3dVector(colors[np.where(labels >= 0)])

    return pcd, bounding_boxes, convex_hulls, triangle_meshes

# Initializations
bin = r"D:\Downloads\00\velodyne\000000.bin"
point_cloud = np.fromfile(bin, dtype=np.float32)
point_cloud = point_cloud.reshape(-1, 4)
points = point_cloud[:, :3]

# Pointcloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Remove large planar areas
pcd = remove_planes(pcd, distance_threshold=0.3)
pcd, _, convex_hulls, triangle_meshes = clustering(pcd, eps=0.6, min_points=5)
o3d.visualization.draw_geometries([ *convex_hulls, *triangle_meshes])