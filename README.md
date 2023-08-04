# 3dPointcloud
Lidar pointcloud from kitti odometry 

The data I used it from the kitti-odometry.
Starting up I took the lidar pointcloud and removed the large planar areas.
Then I used DBSCAN to turn the points of the pointcloud to clusters and used a metric to remove noise points.
Every cluster that remains is an actual object and I show their bounding boxes and their 3d meshes.
