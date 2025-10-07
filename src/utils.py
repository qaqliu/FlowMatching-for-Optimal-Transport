import math
import torch

def get_v(
    x_l,
    x_r,
    t,
    method = 'ot' # 'ot' or 'trig'
):
    if method == 'ot':
        return x_r - x_l
    elif method == 'trig':
        v = - math.pi / 2 * torch.sin(math.pi / 2 * t) * x_l + math.pi / 2 * torch.cos(math.pi / 2 * t) * x_r
        return v
    else:
        raise ValueError("method should be 'ot' or 'trig'")
    
def get_path(
    x_l,
    x_r,
    t,
    method = 'ot' # 'ot' or 'trig'
):
    if method == 'ot':
        return (1 - t) * x_l + t * x_r
    elif method == 'trig':
        return torch.cos(math.pi / 2 * t) * x_l + torch.sin(math.pi / 2 * t) * x_r
    else:
        raise ValueError("method should be 'ot' or 'trig'")
    
def get_mse(
    x_l,
    x_r,
    t,
    model,
    method = 'ot' # 'ot' or 'trig'
):
    v = get_v(x_l, x_r, t, method)
    path = get_path(x_l, x_r, t, method)
    v_hat = model(path, t)
    mse = ((v - v_hat) ** 2)
    return mse.sum(dim=1, keepdim=True) # (B, 1)

def get_loss(
    x_l,
    x_r,
    model,
    method = 'ot', # 'ot' or 'trig'
    int_method = 'Simpson38',
    num_int_pts = 3,
):
    output = torch.zeros(x_l.size(0), 1, device=x_l.device)
    if int_method == 'Simpson38':
        h = 1.0 / num_int_pts
        for i in range(num_int_pts):
            t_now = i * h
            k1 = get_mse(x_l, x_r, t_now, model, method)
            k2 = get_mse(x_l, x_r, t_now + h / 3, model, method)
            k3 = get_mse(x_l, x_r, t_now + 2 * h / 3, model, method)
            k4 = get_mse(x_l, x_r, t_now + h, model, method)
            output += (k1 + 3 * k2 + 3 * k3 + k4) * h / 8
    else:
        raise ValueError("int_method should be 'Simpson38'")
    return output.mean()