#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: On Oversquashing project authors 
"""

import sys
import os
import random
import torch
import torch.nn.functional as F
import git
import numpy as np
import wandb
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data.sampler import SubsetRandomSampler
from parser import get_parser

from torch_geometric.loader import DataLoader

from torch_geometric.data.collate import collate
from torch_geometric.data import Data, InMemoryDataset

from utils.factory import build_model, build_dataset
from utils.utils import set_seed, reset_wandb_env


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)[data.mask]
    loss = F.cross_entropy(out, data.y)
    loss.backward()
    optimizer.step()
    del out


def test(model, data):
    model.eval()
    with torch.no_grad():
        logits, accs, losses, preds = model(data.x, data.edge_index), [], [], []
        mask = data.mask
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y).sum().item() / mask.sum().item()
        loss = F.cross_entropy(logits[mask], data.y)
        preds.append(pred.detach().cpu())
        accs.append(acc)
        losses.append(loss.detach().cpu())
        return accs, losses

def run_exp(args, dataset, model):
    data = dataset
    model = model.to(args['device'])

    assert len(data) == args.synth_train_size + args.synth_test_size
    # 5k for training 
    # 500 for test
    # following Bodnar et al. CW networks
    train_indices = list(range(0, args.synth_train_size))
    test_indices = list(range(args.synth_train_size, len(data)))
    train_data = data[:args.synth_train_size]
    test_data = data[args.synth_train_size:]

    train_loader = DataLoader(train_data,
                              shuffle=True,
                              batch_size=args.bs)


    test_loader = DataLoader(test_data,
                              shuffle=False,
                              batch_size=len(test_indices))

    optimizer = torch.optim.Adam(model.parameters(),
                                 weight_decay=args.weight_decay,
                                 lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                    mode="max",
                    factor=0.5,
                    patience=10,
                    min_lr=1e-5,
                    verbose=True)


    epoch = 0
    best_test_acc = 0.0
    best_test_loss = float('inf')
    best_epoch = 0
    bad_counter = 0
    keep_running = True
    for epoch in range(args['epochs']):
        train_acc, train_loss, test_acc, test_loss = [], [], [], [] 

        #train
        for batch in train_loader:
            train(model, optimizer, batch.to(args.device))
            train_acc_batch, train_loss_batch = test(model, batch.to(args.device))
            train_acc.append(train_acc_batch)
            train_loss.append(train_loss_batch)
  
        #test
        for batch in test_loader:
            test_acc_batch, test_losses_batch = test(model, batch.to(args.device))
            test_acc.append(test_acc_batch)
            test_loss.append(test_losses_batch)
  
  
  
        train_acc = np.mean(train_acc)
        train_loss = np.mean(train_loss)
        test_acc = np.mean(test_acc)
        test_loss = np.mean(test_loss)

  
        res_dict = {
            'train_acc': train_acc,
            'train_loss': train_loss,
            'test_acc': test_acc,
            'test_loss': test_loss,
        }
  
        wandb.log(res_dict, step=epoch)
  
        new_best_trigger = test_acc > best_test_acc if args[
            'stop_strategy'] == 'acc' else test_loss < best_test_loss
        if new_best_trigger:
            best_test_acc = test_acc
            best_test_loss = test_loss
            test_acc = test_acc
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1
  
        if bad_counter == args['early_stopping']:
            keep_running = False if test_acc < args['min_acc'] else True
            break

        if test_acc == 1.0:
            print("Perfect accuracy, stopping to save resources.")
            break
  
        scheduler.step(test_acc)
        print(f"Epochs: {epoch} | Best epoch: {best_epoch}")
        print(f"Test acc: {test_acc:.4f}")
        print(f"Best test acc: {best_test_acc:.4f}")
  
    wandb.log({'best_test_acc': test_acc,
              'best_test_acc': best_test_acc, 'best_epoch': best_epoch})
    keep_running = False if test_acc < args['min_acc'] else True

    return test_acc, best_test_acc, keep_running



if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    reset_wandb_env()

    try:
        model_cls = build_model(args)
    except:
        raise ValueError(f'Unknown model {args.model}')

    args.device = torch.device(
        f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    dataset = build_dataset(args)
    random.shuffle(dataset)
    
    # Add extra arguments
    args.sha = sha
    args.hidden_channels = args.hidden_dim
    args.num_layers = args.mpnn_layers

    set_seed(args.seed)

    results = []
    wandb.init(project="on-oversquashing", config=vars(args), entity=args.entity)
    print(args)

    test_acc, test_loss, keep_running = run_exp(
        wandb.config, dataset, model_cls)
    results.append([test_acc, test_loss])
    test_acc_mean, test_loss_mean = np.mean(results, axis=0) * 100.0
    test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100.0


    wandb_results = {'final_loss': test_loss_mean,
                 'final_acc': test_acc_mean, 'final_acc_std': test_acc_std}
    wandb.log(wandb_results)
    wandb.finish()

    model_name = args.model
    print(f'{model_name} on {args.dataset} | SHA: {sha}')
    print(
        f'Test acc: {test_acc_mean:.4f} +/- {test_acc_std:.4f} | Test loss: {test_loss_mean:.4f}')