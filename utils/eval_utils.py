import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import plotly.express as px
import pandas as pd
from sklearn.neighbors import NearestNeighbors, KDTree
import open3d as o3d
import os.path as osp
import os
import json
import shutil

def build_gallery(embedder: torch.nn.Module,
                  loader: DataLoader,
                  device: torch.device) -> np.ndarray:
    """Builds gallery using the images in the dataloader

    Args:
        embedder: the model to use to get the embedding.
        loader: the dataloader containing the dataset on which build the gallery.
        device: the device to use for the model.

    Returns:
        The gallery, i.e. an array with all the embeddings for the images
        in the dataset.
    """
    gallery = []
    with torch.no_grad():
        for points, _ in tqdm(loader):
            points = torch.Tensor(points).transpose(2, 1)
            points = points.to(device)
            batch_embeddings = embedder(points)
            gallery.extend(batch_embeddings.to('cpu').numpy())

    return np.asarray(gallery)

def visualize_embeddings(ids:np.array, embeddings_reduced:np.array, labels:np.array):
  '''
  Visualize reduced embeddings by umap algorithm

  Args:
      ids: numpy array containing the filenames of the 3d objects
      embeddings_reduced: reduced embeddings representing the 3d objects through umap
      labels: category of each 3d object
  '''
  df = pd.DataFrame({'id':ids,'x':embeddings_reduced[:,0], 'y':embeddings_reduced[:,1], 'category':labels})
  fig = px.scatter(df, x="x", y="y", color="category", hover_data=['id'])
  fig.show()

def nearest_neighbor(sources: np.ndarray,
                     source_idx: int,
                     targets: np.ndarray,
                     num_neighbors: int,
                     algorithm: str = 'kd_tree') -> tuple[np.ndarray, np.ndarray]:
    """Computes nearest neighbor search.

    Estimates for each sample in source the nearest neighbor in target using
    the specified algorithm.

    Args:
        sources: the source samples.
        targets: the target samples.
        num_neighbors: the number of neighbors to find.
        algorithm: the algorithm to use kd_tree or brute force.

    Returns:
        The euclidean distance from each sample in source to the nearest neighbor in target.
        The indices of the nearest neighbor points on target for each sample in source.
    """
    kd_tree = NearestNeighbors(n_neighbors=num_neighbors+1,
                               algorithm=algorithm,
                               metric='euclidean')
    kd_tree.fit(targets)
    distances, indices = kd_tree.kneighbors(sources)

    return distances[indices != source_idx], indices[indices != source_idx]

def draw_geometries(geometries,title=''):
    graph_objects = []

    for geometry in geometries:
        geometry_type = geometry.get_geometry_type()

        if geometry_type == o3d.geometry.Geometry.Type.PointCloud:
            points = np.asarray(geometry.points)
            colors = None
            if geometry.has_colors():
                colors = np.asarray(geometry.colors)
            elif geometry.has_normals():
                colors = (0.5, 0.5, 0.5) + np.asarray(geometry.normals) * 0.5
            else:
                geometry.paint_uniform_color((1.0, 0.0, 0.0))
                colors = np.asarray(geometry.colors)

            scatter_3d = go.Scatter3d(x=points[:,0], y=points[:,1], z=points[:,2], mode='markers', marker=dict(size=1, color=colors))
            graph_objects.append(scatter_3d)

        if geometry_type == o3d.geometry.Geometry.Type.TriangleMesh:
            triangles = np.asarray(geometry.triangles)
            vertices = np.asarray(geometry.vertices)
            colors = None
            if geometry.has_triangle_normals():
                colors = (0.5, 0.5, 0.5) + np.asarray(geometry.triangle_normals) * 0.5
                colors = tuple(map(tuple, colors))
            else:
                colors = (1.0, 0.0, 0.0)

            mesh_3d = go.Mesh3d(x=vertices[:,0], y=vertices[:,1], z=vertices[:,2], i=triangles[:,0], j=triangles[:,1], k=triangles[:,2], facecolor=colors, opacity=0.50)
            graph_objects.append(mesh_3d)

    fig = go.Figure(
        data=graph_objects,
        layout=dict(
            title=dict(text=title),
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)
            )
        )
    )
    fig.show()

def read_mesh(path:str) -> o3d.geometry.TriangleMesh:
  '''Reads triangle mesh and computes its vertex normals.

  Args:
      path: path from which the mesh is read

  Returns:
      o3d.geometry.TriangleMesh'''
  mesh = o3d.io.read_triangle_mesh(path)
  mesh.compute_vertex_normals()
  return mesh

def show_neighbors(query_idx,
                   indices: np.ndarray,
                   distances: np.ndarray,
                   dataset: Dataset,
                   threshold: float,
                   data_root: str,
                   class_query: str = None) -> None:
    """Shows the query image together with the found nearest neighbors.

    Args:
        query_pc: the input point cloud.
        indices: the indices of nearest neighbors.
        distances: the distances of nearest neighbors.
        dataset: the dataset on which the gallery was built.
        threshold: the matching threshold
        data_root: directory where the mesh data is located
        class_query: the name of the class of the query image.

    Returns:
    """
    o3d.visualization.draw_geometries = draw_geometries # replace function
    mesh_dir = f'{data_root}/mesh'
    translation = (1,0,0)
    
    class_query_title = class_query if class_query else ''

    # Read corresponding mesh format and translate
    query_mesh_filename = dataset.samples[query_idx][0].replace('.pcd','.off')
    query_mesh = read_mesh(osp.join(mesh_dir, query_mesh_filename))

    pcd = query_mesh.sample_points_uniformly(number_of_points=1024).translate(translation)

    o3d.visualization.draw_geometries([query_mesh],f'{class_query_title} ({query_mesh_filename})')

    classes_map = {v:k for k,v in dataset.classes.items()}

    for idx, (idx_nn, dist_nn) in enumerate(zip(indices, distances)):
        color = np.random.rand(3)
        points, label_matched = dataset[idx_nn]
        class_match = classes_map[label_matched]

        if class_query is not None:
            symbol = '✅' if class_query == class_match else '❌'
        else:
            symbol = '✅' if dist_nn <= threshold else '❌'

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Read corresponding mesh format and translate
        mesh_filename = dataset.samples[idx_nn][0].replace('.pcd','.off')
        mesh = read_mesh(osp.join(mesh_dir, mesh_filename))
        pcd = mesh.sample_points_uniformly(number_of_points=1024).translate(translation)

        o3d.visualization.draw_geometries([mesh], f'{symbol} {class_match} \n Distance: {dist_nn:.3f} ({mesh_filename})')

def get_recalls(model: torch.nn.Module,
                dataset: Dataset,
                gallery: np.ndarray,
                kk: list,
                exp_name:str,
                device:str) -> dict:
    """Computes the recall using different nearest neighbors searches.

    Args:
        model: the model to use to compute the embedding.
        dataset: the dataset to use to compute the recall.
        gallery: the gallery containing all the embeddings for the dataset.
        kk: the number of nearest neighbors to use for each recall.

    Returns:
        The computed recalls with different nearest neighbors searches.
    """
    model.eval().to(device)
    max_nn = max(kk)
    recalls = {idx: 0. for idx in kk}
    targets = np.asarray(dataset.targets)
    tree = KDTree(gallery)

    # Create a numpy array of datapaths (e.g. tree/M001285.pcd)
    datapaths = np.array(list(map(lambda x: x[0], dataset.samples)))
    root = osp.join('drive/MyDrive/ShapeRetrieval/results', exp_name, 'inference')
    if not osp.exists(root):
      os.mkdir(root)
    else:
      shutil.rmtree(root)

    for i, (points, label_query) in enumerate(tqdm(dataset)):
        points = torch.Tensor(points).unsqueeze(0).to(device)
        points = points.transpose(2, 1)

        category = dataset.samples[i][1]
        if not osp.exists(osp.join(root, category)): os.mkdir(osp.join(root, category))

        with torch.no_grad():
            embedding = model(anchor=points)
            embedding = embedding.to('cpu').numpy()

            _, indices_matched = tree.query(embedding, k=max_nn + 1)
            indices_matched = indices_matched[0]

            # Save matched neighbours for a single sample in a .txt file
            filename = osp.basename(datapaths[i]).replace('.pcd','.txt')
            filepath = osp.join(root, category, filename)
            np.savetxt(filepath, datapaths[indices_matched], fmt='%s')

            for k in kk:
                indices_matched_temp = indices_matched[1:k + 1]
                classes_matched = targets[indices_matched_temp]
                recalls[k] += np.count_nonzero(classes_matched == label_query) > 0

    for key, value in recalls.items():
        recalls[key] = value / (1.0 * len(dataset))

    print(f"\nSaved inference results in {root}")

    return recalls

def save_load_results(directory:str, data:tuple=None, save:bool=True) -> tuple:
  '''
  Saves or loads results depending on the 'save' variable

  Args:
      directory: root directory in which to save/load results
      data: contains recall scores dictionary and the gallery
      save: bool which specifies whether data should be saved or loaded
  Returns:
      recall scores dictionary and gallery when 'save' is False
  '''
  recalls_filename = "point_tnt_recall.json"
  gallery_filename = "gallery_shrec.txt"

  if save:
    if data is not None:
      recalls_with_triplet, gallery_shrec = data
      with open(osp.join(directory, recalls_filename), "w") as f:
        json.dump(recalls_with_triplet, f)
      np.savetxt(osp.join(directory, gallery_filename), gallery_shrec, fmt='%f')
      return None
  else:
    with open(osp.join(directory, recalls_filename)) as f:
      recalls_with_triplet = json.load(f)
    gallery_shrec = np.loadtxt(osp.join(directory, gallery_filename), dtype=float)
    return recalls_with_triplet, gallery_shrec