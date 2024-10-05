import open3d as o3d
import os.path as osp
import os
import glob
import numpy as np

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def main():
    p = r"C:\Users\zoril\Downloads\mesh\airplane"
    filepaths = glob.glob(p + "/*.off")
    print(len(filepaths))
    meshes = []
    for i,fp in enumerate(filepaths[:5]):
        mesh = o3d.io.read_triangle_mesh(fp)
        vertices = pc_normalize(np.asarray(mesh.vertices))
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh = mesh.translate((i*2,0,0))
        mesh.compute_vertex_normals()
        meshes.append(mesh)

    o3d.visualization.draw_geometries(meshes)

if __name__ == "__main__":
    main()