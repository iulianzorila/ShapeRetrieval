import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import os.path as osp
from utils.eval_utils import nearest_neighbor, show_neighbors, build_gallery, save_load_results
from model.embedder import Embedder
from shrec_dataset import ShrecDataset
import random
import argparse

def fix_random(seed: int) -> None:
    """Fix all the possible sources of randomness.

    Args:
        seed: the seed to use.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def query_model(test_dataset:Dataset, df_test:pd.DataFrame, gallery_shrec:np.array, model:torch.nn.Module, category:str, k:int, device:str):
  '''
  Queries the model, by showing the most k similar objects to the given one

  Args:
      test_dataset: unseen data from which the query will be taken
      df: dataframe created from test samples
      gallery_shrec: gallery, i.e. embedded vectors created from the test set
      model: embedder pytorch model in evaluation state
      category: category from which the query will be taken
      k: number of output neighbors to the given query
      device: CUDA or cpu, on which the query will be evaluated
  '''
  fix_random(42)
  index = df_test.groupby('category').get_group(category).index[0]

  points, label_query = test_dataset[index]
  original_points = points.copy()
  points = torch.Tensor(points)
  points = points.unsqueeze(0).transpose(2, 1).to(device)
  with torch.no_grad():
      embedding_points = model(anchor=points)
      embedding_points = embedding_points.to('cpu').numpy()

  nn_distances, nn_indices = nearest_neighbor(embedding_points,
                                              index,
                                              gallery_shrec,
                                              k)

  classes_map = {v:k for k,v in test_dataset.classes.items()}
  show_neighbors(index,
                nn_indices,
                nn_distances,
                test_dataset,
                0.85,
                classes_map[label_query])
  

def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--data_root', type=str, help='folder where point cloud data is located')
    parser.add_argument('--exp_dir', type=str, help='folder from where to load model weights')
    parser.add_argument('--category', type=str, default="chair", help="category on which shape retrieval will be performed. One of the following: \
                        ['chair', 'table', 'armchair', 'airplane', 'sword', 'tree', 'cabinet', 'fish', 'bookshelf', 'head', 'laptop', 'guitar', \
                        'train', 'motorbike', 'cell_phone', 'potted_plant', 'bed', 'helicopter', 'knife', 'keyboard', 'piano', 'couch', 'bus', \
                        'floor_lamp', 'wheel', 'ship', 'house', 'bicycle', 'face', 'door', 'horse', 'flying_bird', 'hand', 'eyeglasses', 'tablelamp', \
                        'standing_bird', 'vase', 'spoon', 'flower_with_stem', 'telephone', 'spider', 'race_car', 'violin', 'ant', 'satellite', 'bench', \
                        'truck', 'car_sedan', 'rifle', 'submarine', 'computer_monitor', 'dolphin'] ")
    parser.add_argument('--num_neighbors', type=int, default=3, help='number of neighbors to retrieve')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size in training')
    parser.add_argument('--num_workers', type=int, default=2, help='number of used workers for dataloader')
    parser.add_argument('--embeddings_dim', type=int, default=512, help='dimension of the generated embeddings')
    return parser.parse_args()

def main():
    args = parse_args()
    model_weights = "best_model.pth" # @param ["best_model.pth", "model.pth"]

    model_path = f'{args.exp_name}/weights/{model_weights}'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = Embedder(size_embedding=args.embeddings_dim, 
                        normalize_embedding=True, 
                        train_feature_extactor=True)
    
    embedder.to(device)

    #checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    print("\n Loading embedder weights \n")
    checkpoint = torch.load(model_path)
    embedder.load_state_dict(checkpoint['model_state_dict'])

    test_dataset = ShrecDataset(args.data_root, 'test')
    sampler = torch.utils.data.SequentialSampler(test_dataset)
    test_gallery_loader = DataLoader(test_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    pin_memory=True,
                                    num_workers=args.num_workers,
                                    sampler=sampler)

    # Create dataframe for query evaluation
    df_test = pd.DataFrame(test_dataset.samples, columns=['filename','category'])

    embedder = embedder.eval()

    if osp.exists(osp.join(args.exp_name, "gallery_shrec.txt")):
        recalls_with_triplet, gallery_shrec = save_load_results(args.exp_name, save=False)
        print(f"\nLoaded gallery from {osp.join(args.exp_name, 'gallery_shrec.txt')} \n")
    else:
        gallery_shrec = build_gallery(embedder, test_gallery_loader, device)
        print(f'The SHREC gallery has: {len(gallery_shrec)} samples')

    query_model(test_dataset,
                df_test,
                gallery_shrec,
                embedder,
                args.category,
                args.num_neighbors)
    
if __name__ == '__main__':
    main()