import numpy as np
from torch.utils.data import Dataset
import os.path as osp
import open3d as o3d
import random
import gdown
import os
import zipfile

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class ShrecDataset(Dataset):
    def __init__(self, root, split='train', filename_prefix=''):
        self.root = root

        if not osp.exists(root + "/pointcloud"):
            print(f"Data was not found in {root + '/pointcloud'}\n")
            self.download_data()

        self.root += "/pointcloud"

        # List of (file path, category)
        self.samples = [(line.rstrip(), line.split('/')[0])
                        for line in open(osp.join(self.root, f'{filename_prefix}{split}.txt'))]

        # Get total categories and map to integers
        self.cat = np.unique(list(map(lambda dp: dp[1], self.samples)))
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        self.targets = np.array([self.classes[dp[1]] for dp in self.samples])

        assert (split == 'train' or split == 'test')
        print('The size of %s data is %d' % (split, len(self.samples)))

    def download_data(self):
        """Download pointcloud SHREC data and unzip to data/pointcloud """
        print("Downloading pointcloud data to data/pointcloud ...")
        url = "https://drive.google.com/uc?id=1oyK3OwVhYMW29ht7hSCN3HUB6QNLOKyU"
        gdown.download(url, self.root + "pointcloud.zip")
        
        with zipfile.ZipFile(self.root + "pointcloud.zip", 'r') as zip_ref:
            zip_ref.extractall(self.root)

        print("Done!")

    def __len__(self):
        return len(self.samples)

    def _get_item(self, index):
        data_path, category = self.samples[index]
        label = self.classes[category]

        pcd = o3d.io.read_point_cloud(osp.join(self.root, data_path))
        points = np.asarray(pcd.points).astype(np.float32)

        points = pc_normalize(points)

        return points, label

    def __getitem__(self, index):
        return self._get_item(index)

class ShrecTripletDataset(ShrecDataset):
  def _get_item(self, index):

      # Get the anchor
      index_anchor = index
      path_anchor, class_anchor = self.samples[index]
      label_anchor = self.classes[class_anchor]
      mask = self.targets == label_anchor
      indices_positive = np.nonzero(mask)[0]

      # Get the positive
      index_positive = index_anchor
      # Generate random number until it's different from index_anchor
      while index_positive == index_anchor:
          index_positive = np.random.choice(indices_positive)
      path_positive, class_positive = self.samples[index_positive]
      label_positive = self.classes[class_positive]

      # Get the negative
      indices_negatives = set(np.arange(len(self.samples))) - set(indices_positive)
      index_negative = random.choice(list(indices_negatives))
      path_negative, class_negative = self.samples[index_negative]
      label_negative = self.classes[class_negative]

      if not label_anchor == label_positive:
          raise ValueError(f'Anchor and Positive from different class: {class_anchor} - {class_positive}')

      if label_anchor == label_negative:
          raise ValueError(f'Anchor and Negative from the same class: {class_anchor} - {class_negative}')

      anchor_path = osp.join(self.root, self.samples[index_anchor][0])
      anchor = np.asarray(o3d.io.read_point_cloud(anchor_path).points).astype(np.float32)

      positive_path = osp.join(self.root, self.samples[index_positive][0])
      positive = np.asarray(o3d.io.read_point_cloud(positive_path).points).astype(np.float32)

      negative_path = osp.join(self.root, self.samples[index_negative][0])
      negative = np.asarray(o3d.io.read_point_cloud(negative_path).points).astype(np.float32)

      anchor = pc_normalize(anchor)
      positive = pc_normalize(positive)
      negative = pc_normalize(negative)

      return (anchor, positive, negative), [label_anchor, label_positive, label_negative]