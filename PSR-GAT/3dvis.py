import open3d as o3d
import os
pcd = o3d.io.read_point_cloud(os.getcwd()+'/model/new/result/newpointcloud/pointcloud/00000000.xyz', print_progress=True)
o3d.visualization.draw_geometries([pcd])
'''
读取点云文件：
read_point_cloud(filename, format=‘auto’, remove_nan_points=True, remove_infinite_points=True, print_progress=False)
参数：
     filename（str）：文件路径。
     格式（str，可选，默认='auto'）：输入文件的格式。 如果未指定或设置为``auto''，则从文件扩展名中推断格式。
     remove_nan_points（布尔型，可选，默认为True）：如果为true，则从PointCloud中删除所有包含NaN的点。
     remove_infinite_points（布尔型，可选，默认为True）：如果为true，则从PointCloud中删除所有包含无限值的点。
     print_progress（布尔型，可选，默认= False）：如果设置为true，则在控制台中可视化进度条
     print("Testing IO for meshes ...")
     
读取网格文件：
mesh = o3d.io.read_triangle_mesh("../../TestData/knot.ply")
print(mesh)
o3d.io.write_triangle_mesh("copy_of_knot.ply", mesh)

numpy转换为open3d格式:open3d.utility.Vector3dVector函数
class open3d.utility.Vector3dVector
将形状（n，3）的float64 numpy数组转换为Open3D格式
eg.
pcd = open3d.geometry.PointCloud()
np_points = np.random.rand(100, 3)

# From numpy to Open3D
pcd.points = open3d.utility.Vector3dVector(np_points)

# From Open3D to numpy
np_points = np.asarray(pcd.points)
'''