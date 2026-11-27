import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import load_data

def get_dataloaders(BATCH_SIZE=32, N_WORKERS=12):
    x_train, x_val, x_test, y_train, y_val, y_test = load_data()

    # to tensor form
    x_train_t = torch.tensor(x_train).permute(0, 2, 1)
    x_val_t = torch.tensor(x_val).permute(0, 2, 1)
    x_test_t = torch.tensor(x_test).permute(0, 2, 1)

    y_train_t = torch.tensor(y_train, dtype=torch.long)  # class numbers 0/1/2
    y_val_t   = torch.tensor(y_val, dtype=torch.long)
    y_test_t  = torch.tensor(y_test, dtype=torch.long)

    train_data  = TensorDataset(x_train_t, y_train_t)
    val_data    = TensorDataset(x_val_t, y_val_t)
    test_data   = TensorDataset(x_test_t, y_test_t)
    
    train_dataloader = DataLoader(dataset= train_data,
                                batch_size= BATCH_SIZE,
                                num_workers= N_WORKERS,
                                shuffle= True)
    
    val_dataloader = DataLoader(dataset= val_data,
                                batch_size= BATCH_SIZE,
                                num_workers= N_WORKERS,
                                shuffle= False)
    
    test_dataloader = DataLoader(dataset= test_data,
                                batch_size= BATCH_SIZE,
                                num_workers= N_WORKERS,
                                shuffle= False)
    
    return train_dataloader, val_dataloader, test_dataloader

def train_step(model: nn.Module,
                dataloader: torch.utils.data.DataLoader,
                loss_fn: nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device):
    
    model.to(device)
    model.train()
    train_loss, train_acc = 0, 0
    
    for X, y in DataLoader:
        X, y = X.to(device), y.to(device)
        
        y_logit = model(X)
        
        loss = loss_fn(y_logit, y)
        train_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        y_label = y_logit.argmax(dim=1)
        train_acc += (y_label == y).sum().item() / len(y_label)
        
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc

def val_test_step(model: nn.Module,
                dataloader: torch.utils.data.DataLoader,
                loss_fn: nn.Module,
                device: torch.device):
    
    model.to(device)
    model.eval()
    loss, acc = 0, 0
    
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            y_logit = model(X)
            
            loss = loss_fn(y_logit, y)
            loss += loss.item()
            
            y_label = y_logit.argmax(dim=1)
            acc += (y_label == y).sum().item() / len(y_label)
            
        loss = loss / len(dataloader)
        acc = acc / len(dataloader)
        
    return loss, acc

def train(model: nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        device: torch.device = device):
    
    model.to(device)
    
            
        
    

    
    

