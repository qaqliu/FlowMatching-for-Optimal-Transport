import yaml
import argparse
from argparse import Namespace
from src import get_data, model, training
import torch
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Load hyperparameters from a YAML file.')
parser.add_argument('--JKO_config', default = 'configs/FM_OT.yaml', type=str, help='Path to the YAML file')
args_parsed = parser.parse_args()

with open(args_parsed.JKO_config, "r") as f:
    config = yaml.safe_load(f)

if __name__ == '__main__':
    save_path = config['training']['save_path']
    
    checkpoint = torch.load(save_path, weights_only=False)
    flow = checkpoint['model']
    flow.eval()
    p_test = get_data.inf_train_gen(
        'img_checkerboard.png', 10000
    ).to(device)
    q_test = get_data.gen_moons(10000, device=device)
    
    with torch.no_grad():
        p_test, _, _ = flow.push_forward(p_test, full_traj=True)
        q_test, _, _ = flow.push_forward(q_test, reverse=True, full_traj=True)

    
    fig, axs = plt.subplots(4, 4, figsize=(12, 12))
    for i in range(4):
        for j in range(4):
            axs[i, j].scatter(p_test[4*i+j][:, 0].cpu(), p_test[4*i+j][:, 1].cpu(), s=1, alpha = 0.3, color='blue')
            
    plt.savefig('results/from_p_to_q.png')
    
    fig, axs = plt.subplots(4, 4, figsize=(12, 12))
    for i in range(4):
        for j in range(4):
            axs[i, j].scatter(q_test[4*i+j][:, 0].cpu(), q_test[4*i+j][:, 1].cpu(), s=1, alpha = 0.3, color='red')
            
    plt.savefig('results/from_q_to_p.png')