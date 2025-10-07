import yaml
import argparse
from argparse import Namespace
from src import get_data, model, training, utils
import torch
import wandb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Load hyperparameters from a YAML file.')
parser.add_argument('--JKO_config', default = 'configs/FM_OT.yaml', type=str, help='Path to the YAML file')
args_parsed = parser.parse_args()

with open(args_parsed.JKO_config, "r") as f:
    config = yaml.safe_load(f)

if __name__ == '__main__':
    N = config['data']['N']
    
    p_tensor = get_data.inf_train_gen(
        'img_checkerboard.png', N
    )
    q_tensor = get_data.gen_moons(N)

    ODEFunc_config = config['model']['ODEFunc_config']
    data_dim = config['model']['data_dim']
    h_k = config['model']['h_k']
    h_steps = config['model']['h_steps']
    activation = config['model']['activation']
    
    func = model.ODEFunc(
        device=device,
        data_dim=data_dim,
        layer_config=ODEFunc_config,
        activation=activation,
        use_time=True,
    )
    
    flow = model.Flow_Matching(
        odefunc=func,
        device=device,
        h_k=h_k,
        h_steps=h_steps,
    )
    
    batch_size = config['training']['batch_size']
    lr = config['training']['lr']
    epochs = config['training']['epochs']
    clip_grad = config['training']['clip_grad']
    save_path = config['training']['save_path']
    shuffle = config['training']['shuffle']
    run = wandb.init(project="Flow_DRE", name='FM_OT')
    training.train_FM_OT(
        flow=flow,
        p_tensor=p_tensor,
        q_tensor=q_tensor,
        num_epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        clip_grad=clip_grad,
        save_path=save_path,
        device=device,
        shuffle=shuffle,
    )
    run.finish()