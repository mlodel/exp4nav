import numpy as np
import torch

import cv2


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


def explained_variance(ypred, y):
    """
    *** copied from openai/baselines ***
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary == 0 else 1 - np.var(y - ypred) / vary


def imagenet_rgb_preprocess(imgs, device=torch.device('cpu'), output_rgb=True):
    if not isinstance(imgs, torch.Tensor):
        imgs = torch.from_numpy(imgs).float()

    if output_rgb:
        imgs = torch.stack((imgs,) * 3, dim=-1)
    else:
        imgs = imgs.unsqueeze(dim=-1)

    if len(imgs.shape) == 4:  # Rollout
        imgs = imgs.unsqueeze(dim=0) # Sequence dim

    imgs = imgs.permute(0, 1, 4, 2, 3).to(device)  # (Seq, N, C, H, W)
    # imgs = imgs / 255.0
    if output_rgb:
        rgb_mean = torch.tensor([0.485, 0.456, 0.406], device=device)
        rgb_std = torch.tensor([0.229, 0.224, 0.225], device=device)
        rgb_mean = rgb_mean[None, None, :, None, None]
        rgb_std = rgb_std[None, None, :, None, None]
        imgs = (imgs - rgb_mean) / rgb_std

    return imgs


def imagenet_grayscale_preprocess(imgs, device=torch.device('cpu')):
    imgs = cv2.cvtColor(imgs, cv2.COLOR_GRAY2RGB)
    if not isinstance(imgs, torch.Tensor):
        imgs = torch.from_numpy(imgs).float()
    if len(imgs.shape) == 4:
        imgs = imgs.unsqueeze(dim=0)
    imgs = imgs.permute(0, 1, 4, 2, 3).to(device)  # (Seq, N, C, H, W)
    imgs = imgs / 255.0
    rgb_mean = torch.tensor([0.485, 0.456, 0.406], device=device)
    rgb_std = torch.tensor([0.229, 0.224, 0.225], device=device)
    rgb_mean = rgb_mean[None, None, :, None, None]
    rgb_std = rgb_std[None, None, :, None, None]
    imgs = (imgs - rgb_mean) / rgb_std
    return imgs


def states_preprocess(obs, device=torch.device('cpu')):

    if isinstance(obs, dict):
        state_keys = ['heading_global_frame', 'angvel_global_frame', 'pos_global_frame', 'vel_global_frame']
        # Normalize States
        state_bounds = dict()
        state_bounds['heading_global_frame'] = [-np.pi, np.pi]
        state_bounds['angvel_global_frame'] = [-3.0, 3.0]
        state_bounds['pos_global_frame'] = [np.array([-10.0, -10.0]), np.array([10.0, 10.0])]
        state_bounds['vel_global_frame'] = [np.array([-3.0, -3.0]), np.array([3.0, 3.0])]

        obs_env = []
        for i in range(list(obs.values())[0].size):
            obs_keys = []
            for key in state_keys:
                obs_norm = ((obs[key][i]-state_bounds[key][0]) * 2 / (state_bounds[key][1] - state_bounds[key][0])) - 1.0
                obs_keys.append(obs_norm)
            obs_env.append(np.hstack(obs_keys))
        obs_states = np.stack(obs_env)

        # obs_states = np.stack([np.hstack([obs_uint8[key][i] for key in state_keys])
        #                        for i in range(self.config['num_envs'])])

    else:
        obs_states = obs

    states = obs_states
    if not isinstance(states, torch.Tensor):
        states = torch.from_numpy(states).float()
    if len(states.shape) == 2:
        states = states.unsqueeze(dim=0)

    return states.to(device), obs_states


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """ copied from openai/baselines
        Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


class ExpDecaySchedule(object):
    def __init__(self, lam1=0.1, lam2=0.95):
        """ copied from openai/baselines
        decay in lam1*lam2^t
        """
        self.lam1 = lam1
        self.lam2 = lam2

    def value(self, t):
        return self.lam1 * self.lam2 ** t
