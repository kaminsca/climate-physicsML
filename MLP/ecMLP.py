# Simplified ENERGY CONSERVING MULTI-LAYER PERCEPTRON IMPLEMENTATION

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from climsim_utils.data_utils import * 
from climsim_datapip import climsim_dataset
from climsim_datapip_h5 import climsim_dataset_h5
from tqdm import tqdm
from dataclasses import dataclass, field
import numpy as np
from loss_energy import loss_energy
from mlp import MLP
import mlp as mlp

@dataclass
class Config:
    epochs: int = 50
    batch_size: int = 1024
    learning_rate: float = 0.0001
    mlp_hidden_dims: list[int] = field(default_factory=lambda: [256, 256, 256, 256, 256, 256, 256, 256, 256])
    loss: str = 'mse'
    use_energy_loss: bool = True
    energy_loss_weight: float = 1.0
    mlp_layers: int = 9
    save_path: str = "./save_output"
    data_path: str = "/scratch/thore_root/thore0/alvarovh/large_data/cse598_project/subsampled_low_res/" # avh greatlakes
    val_input: str = 'val_input.npy'
    val_target: str = 'val_target.npy'
    train_target: str = "train_target.npy"
    train_input: str = "train_input.npy"
    qc_lbd: str = 'inputs/qc_exp_lambda_large.txt'
    qi_lbd: str = 'inputs/qi_exp_lambda_large.txt'    
    CLIMSIM_REPO_PATH: str = "/home/alvarovh/code/cse598_climate_proj/ClimSim/"


cfg = Config()

grid_path = cfg.CLIMSIM_REPO_PATH + 'grid_info/ClimSim_low-res_grid-info.nc'
norm_path = cfg.CLIMSIM_REPO_PATH + 'preprocessing/normalizations/'

grid_info = xr.open_dataset(grid_path)
input_mean = xr.open_dataset(norm_path + 'inputs/input_mean.nc')
input_max = xr.open_dataset(norm_path + 'inputs/input_max.nc')
input_min = xr.open_dataset(norm_path + 'inputs/input_min.nc')
output_scale = xr.open_dataset(norm_path + 'outputs/output_scale.nc')

lbd_qc = np.loadtxt(norm_path + cfg.qc_lbd, delimiter=',')
lbd_qi = np.loadtxt(norm_path + cfg.qi_lbd, delimiter=',')

data = data_utils(grid_info = grid_info, 
                input_mean = input_mean, 
                input_max = input_max, 
                input_min = input_min, 
                output_scale = output_scale)
data.set_to_v1_vars()
input_size = data.input_feature_len
output_size = data.target_feature_len
input_sub, input_div, out_scale = data.save_norm(write=False)
# Create dataset instances
val_input_path = cfg.data_path + cfg.val_input
val_target_path = cfg.data_path + cfg.val_target
train_input_path = cfg.data_path + cfg.train_input
train_target_path = cfg.data_path + cfg.train_target

if not os.path.exists(cfg.data_path + cfg.val_input):
    raise ValueError('Validation input path does not exist')

val_dataset = climsim_dataset(val_input_path, val_target_path, input_sub, input_div, out_scale, lbd_qc, lbd_qi)
val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, sampler=None)
train_dataset = climsim_dataset(train_input_path, train_target_path, input_sub, input_div, out_scale, lbd_qc, lbd_qi)
train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=False, sampler=None, drop_last=True, pin_memory=torch.cuda.is_available())


# Create model
model = MLP(in_dims=input_size, out_dims=output_size, hidden_dims=cfg.mlp_hidden_dims, layers=cfg.mlp_layers).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Create optimizer
optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

# Create loss function
criterion = nn.MSELoss()
if cfg.loss == 'mse':
    criterion = nn.MSELoss()
elif cfg.loss == 'mae':
    criterion = nn.L1Loss()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hyai = data.grid_info['hyai'].values
hybi = data.grid_info['hybi'].values
hyai = torch.tensor(hyai, dtype=torch.float32).to(device)
hybi = torch.tensor(hybi, dtype=torch.float32).to(device)
out_scale_device = torch.tensor(out_scale, dtype=torch.float32).to(device)

def training_step(model, data_input, target):
    output = model(data_input)
    loss = criterion(output, target)
    return loss
# Training loop
for epoch in range(cfg.epochs):
    model.train()
    train_loss = 0.0
    for data_input, target in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
        data_input, target = data_input.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data_input)
        if cfg.use_energy_loss:
            ps_raw = data_input[:,1500]*input_div[1500]+input_sub[1500]
            loss_energy_train = loss_energy(output, target, ps_raw, hyai, hybi, out_scale_device)*cfg.energy_loss_weight
            loss_orig = criterion(output, target)
            loss = loss_orig + loss_energy_train
        else:
            loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader)}")

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data_input, target in tqdm(val_loader, desc=f'Epoch {epoch+1} [Validation]'):
            data_input, target = data_input.to(device), target.to(device)
            output = model(data_input)
            if cfg.use_energy_loss:
                ps_raw = data_input[:,1500]*input_div[1500]+input_sub[1500]
                loss_energy_train = loss_energy(output, target, ps_raw, hyai, hybi, out_scale_device)*cfg.energy_loss_weight
                loss_orig = criterion(output, target)
                loss = loss_orig + loss_energy_train
            else:
                loss = criterion(output, target)
            val_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Validation Loss: {val_loss/len(val_loader)}")

# Save the model
torch.save(model.state_dict(), f"{cfg.save_path}/model.pth")

print("Training complete!")