from typing import Callable
from torch import nn
import torch
from model.kp_model import KPDetector
from edgedataset import EdgeDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
import os


def checkpoint_save(model: nn.Module, optimizer: Adam):
    torch.save(model.state_dict(), 'model.pth')
    torch.save(optimizer.state_dict(), 'optimizer.pth')


def train(model: KPDetector, optimizer: Adam, dl: DataLoader, loss_fn: Callable, max_epoch: int):
    model.train()
    for epoch in range(max_epoch):
        # print('epoch:', epoch+1, 'lr:', lr_scheduler.get_last_lr()[0])
        print('epoch:', epoch+1)
        bar = tqdm(dl)
        for step, (x, y) in enumerate(bar):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            bar.set_description(
                f'loss:{loss.item():0.4f}')
        checkpoint_save(model, optimizer)
        # lr_scheduler.step()


def euclidean_distance(pred: torch.tensor, y: torch.tensor):
    return torch.norm(pred-y, dim=1).mean()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = EdgeDataset('data.csv')
    dataloader = DataLoader(train_dataset, batch_size=4, num_workers=2,
                            shuffle=True, drop_last=True)
    model = KPDetector(pad=3).to(device)
    lr = 1e-4
    if os.path.exists('model.pth'):
        model.load_state_dict(torch.load('model.pth'))
    optimizer = Adam(model.parameters(), lr=lr)
    if os.path.exists('optimizer.pth'):
        optimizer.load_state_dict(torch.load('optimizer.pth'))
    loss_fn = nn.L1Loss().to(device)
    # loss_fn = euclidean_distance
    # lr_scheduler = MultiStepLR(optimizer, milestones=[
    #     10, 15, 18], gamma=0.1)
    train(model, optimizer, dataloader, loss_fn, 100)
