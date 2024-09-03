from torch.utils.data import DataLoader
from torchsummary import summary
import argparse
import torch
from warmup_scheduler import GradualWarmupScheduler
import os
import os.path as osp
from shrec_dataset import ShrecTripletDataset
from model.embedder import Embedder
from utils.logger import IOStream
from utils.training_utils import training_loop

def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--data_root', type=str, help='folder where point cloud data is located')
    parser.add_argument('--exp_dir', type=str, help='folder where to save training weights and results')
    parser.add_argument('--batch_size', type=int, default=48, help='batch size in training')
    parser.add_argument('--num_workers', type=int, default=2, help='number of used workers for dataloader')
    parser.add_argument('--embeddings_dim', type=int, default=512, help='dimension of the generated embeddings')
    parser.add_argument('--epoch', default=100, type=int, help='number of epoch in training')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    return parser.parse_args()

def main():
    args = parse_args()

    train_loader = DataLoader(ShrecTripletDataset(args.data_root, 'train'),
                            batch_size=args.batch_size,
                            shuffle=True,
                            pin_memory=True,
                            num_workers=args.num_workers)

    test_loader = DataLoader(ShrecTripletDataset(args.data_root, 'test'),
                            batch_size=args.batch_size,
                            pin_memory=True,
                            num_workers=args.num_workers)

    print(f'Batches train: {len(train_loader)} - val: {len(test_loader)}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedder = Embedder(size_embedding=args.embeddings_dim, 
                        normalize_embedding=True, 
                        train_feature_extactor=True)
    embedder.to(device)
    summary(embedder, input_size=(3,1024))

    # Apply weight decay to all layers, except biases, LayerNorm and BatchNorm
    param_optimizer = list(embedder.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'BatchNorm']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.1},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    # Select optimizer
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)

    # Use cosine annealing learning rate scheduler with warmup
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr/100.0)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=scheduler)

    if not osp.exists(args.exp_dir): os.makedirs(args.exp_dir)

    savedir = osp.join(args.exp_dir, 'weights')
    if not osp.exists(savedir): os.makedirs(savedir)

    logdir = osp.join(args.exp_dir,'log')
    if not osp.exists(logdir): os.makedirs(logdir)

    training_loop(IOStream(osp.join(logdir,'run.log')),
                savedir,
                args.epochs,
                optimizer,
                scheduler_warmup,
                5,
                embedder,
                torch.nn.TripletMarginLoss(margin=0.30),
                train_loader,
                test_loader)