import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import time
from model import ODEFunc, Flow_Matching
import utils
import os
import matplotlib.pyplot as plt
import wandb
import math

def train_FM_OT(
    flow: Flow_Matching,
    p_tensor,
    q_tensor,
    num_epochs,
    lr,
    batch_size,
    clip_grad,
    save_path,
    device,
    shuffle=True,
):
    p_tensor, q_tensor = p_tensor.to(device), q_tensor.to(device)
    dataset = TensorDataset(p_tensor, q_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    flow.train()
    for epoch in range(num_epochs):
        
        loss_sum, n = 0.0, 0
        time_start = time.time()
        for x_l, x_r in dataloader:
            optimizer.zero_grad()
            loss = utils.get_loss(x_l, x_r, flow.model, method='ot', int_method='Simpson38', num_int_pts=flow.h_steps)
            loss.backward()
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(flow.parameters(), clip_grad)
            optimizer.step()
            loss_sum += loss.item() * x_l.size(0)
            n += x_l.size(0)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_sum / n:.5f}, Time: {time.time() - time_start:.2f}s")
        wandb.log({"loss": loss_sum / n, "epoch": epoch})
        
    sdict = {
        "model": flow,
        "optimizer": optimizer.state_dict(),
    }
    torch.save(sdict, save_path)
    print(f"Model saved to {save_path}")