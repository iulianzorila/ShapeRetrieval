import torch
from torch.utils.data import DataLoader
from .augmentation import data_aug
from .logger import IOStream
from tqdm import tqdm
import os.path as osp
from timeit import default_timer as timer

# Train one epoch
def train(writer: IOStream,
          model: torch.nn.Module,
          train_loader: DataLoader,
          device: torch.device,
          optimizer: torch.optim,
          criterion,
          log_interval: int,
          epoch: int):
    """Trains a neural network for one epoch.

    Args:
        model: the model to train.
        train_loader: the data loader containing the training data.
        device: the device to use to train the model.
        optimizer: the optimizer to use to train the model.
        criterion: the loss to optimize.
        log_interval: the log interval.
        epoch: the number of the current epoch.

    Returns:
        the loss value on the training data.
    """
    samples_train = 0
    loss_train = 0
    size_ds_train = len(train_loader.dataset)
    num_batches = len(train_loader)

    model.train()
    for idx_batch, (triplet, labels) in tqdm(enumerate(train_loader)):

        anchors, positives, negatives = data_aug(triplet[0],triplet[1],triplet[2])

        anchors = anchors.to(device)
        positives = positives.to(device)
        negatives = negatives.to(device)

        optimizer.zero_grad()
        embedding_anc, embedding_pos, embedding_neg = model(anchors,
                                                            positives,
                                                            negatives)

        loss = criterion(embedding_anc, embedding_pos, embedding_neg)
        loss_train += loss.item() * len(anchors)
        samples_train += len(anchors)

        loss.backward()
        optimizer.step()

        if log_interval > 0:
            if idx_batch % log_interval == 0:
                running_loss = loss_train / samples_train
                global_step = idx_batch + (epoch * num_batches)
                #print('Running loss:', running_loss, 'Global step:',global_step)
                #writer.cprint('Metrics/Loss_Train_IT', running_loss, global_step)

    loss_train /= samples_train
    return loss_train

# Validate one epoch
def validate(model: torch.nn.Module,
             data_loader: DataLoader,
             device: torch.device,
             criterion):
    """Evaluates the model.

    Args:
        model: the model to evalaute.
        data_loader: the data loader containing the validation or test data.
        device: the device to use to evaluate the model.
        criterion: the loss function.

    Returns:
        the loss value on the validation data.
    """
    samples_val = 0
    loss_val = 0.

    model = model.eval()
    with torch.no_grad():
        for idx_batch, (triplet, labels) in tqdm(enumerate(data_loader)):
            anchors = torch.Tensor(triplet[0]).transpose(2, 1)
            positives = torch.Tensor(triplet[1]).transpose(2, 1)
            negatives = torch.Tensor(triplet[2]).transpose(2, 1)

            anchors = anchors.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)

            embedding_anc, embedding_pos, embedding_neg = model(anchors,
                                                                positives,
                                                                negatives)

            loss = criterion(embedding_anc, embedding_pos, embedding_neg)
            loss_val += loss.item() * len(triplet)
            samples_val += len(triplet)

    loss_val /= samples_val
    return loss_val

def training_loop(writer:IOStream,
                  savedir: str,
                  num_epochs: int,
                  optimizer: torch.optim,
                  lr_scheduler: torch.optim.lr_scheduler,
                  log_interval: int,
                  model: torch.nn.Module,
                  criterion: torch.nn.Module,
                  loader_train: DataLoader,
                  loader_val: DataLoader,
                  device:str,
                  verbose: bool=True):
    """Executes the training loop.

        Args:
            writer: logs training results.
            savepath: path to save model weights when improving,
            num_epochs: the number of epochs.
            optimizer: the optimizer to use.
            lr_scheduler: the scheduler for the learning rate.
            log_interval: intervall to print on tensorboard.
            model: the mode to train.
            criterion: the loss to minimize.
            loader_train: the data loader containing the training data.
            loader_val: the data loader containing the validation data.
            device: specifies on which device to train the model ("cpu" or "cuda")
            verbose: if true print the value of loss.

        Returns:
            A dictionary with the statistics computed during the train:
            the values for the train loss for each epoch.
            the values for the validation loss for each epoch.
            the time of execution in seconds for the entire loop.
    """
    loop_start = timer()

    losses_values_train = []
    losses_values_val = []
    best_loss_val = 1e6
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        time_start = timer()
        loss_train = train(writer, model, loader_train, device, optimizer, criterion, log_interval, epoch)
        loss_val = validate(model, loader_val, device, criterion)
        time_end = timer()

        losses_values_train.append(loss_train)
        losses_values_val.append(loss_val)

        lr = optimizer.param_groups[0]['lr']

        if loss_val < best_loss_val:
          best_loss_val = loss_val
          best_epoch = epoch + 1
          state = {
                    'epoch': best_epoch,
                    'best_loss_val': best_loss_val,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }
          print(f'Saving best model at {savedir}/best_model.pth')
          torch.save(state, osp.join(savedir,'best_model.pth'))

        if verbose:
            print(f'Epoch: {epoch} '
                  f' Lr: {lr:.8f} '
                  f' Loss: Train = [{loss_train:.4f}] - Val = [{loss_val:.4f}] - Best Val = [{best_loss_val:.4f}]'
                  f' Time one epoch (s): {(time_end - time_start):.4f} ')

        # Log results
        writer.log_training([epoch, lr, loss_train, loss_val, best_loss_val])

        torch.save(state, osp.join(savedir,'model.pth'))

        # Increases the internal counter
        if lr_scheduler:
            lr_scheduler.step()

    loop_end = timer()
    time_loop = loop_end - loop_start
    if verbose:
        print(f'Time for {num_epochs} epochs (s): {(time_loop):.3f}')

    return {'loss_values_train': losses_values_train,
            'loss_values_val': losses_values_val,
            'time': time_loop}