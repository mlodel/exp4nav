import time
from collections import deque

import numpy as np
import torch
from torch import nn
from torch import optim

from algos.base import BaseAgent
from models.ppo_nets_ig import PPONetsIG
from utils import logger
from utils.color_print import *
from utils.common import imagenet_rgb_preprocess, imagenet_grayscale_preprocess, states_preprocess, safemean


class PPOAgentIG(BaseAgent):
    def __init__(self, env, args, val_env=None):
        super().__init__(env, args, val_env)
        assert self.num_steps % self.rnn_seq_len == 0
        self.net_model = PPONetsIG(act_dim=env.action_space.shape[0],
                                   device=args.device,
                                   fix_cnn=args.fix_cnn,
                                   rnn_type=args.rnn_type,
                                   rnn_hidden_dim=args.rnn_hidden_dim,
                                   rnn_num=args.rnn_num)

        self.net_model.to(self.device)

        self.optimizer = optim.Adam(self.net_model.parameters(),
                                    lr=args.lr,
                                    weight_decay=args.weight_decay)
        self.val_loss_criterion = nn.SmoothL1Loss().to(args.device)

        if args.resume or args.test or args.il_pretrain:
            self.load_model(step=args.resume_step)

        self.reward_counter = np.zeros(self.config['num_envs'])
        self.reward_ig_counter = np.zeros(self.config['num_envs'])
        self.eps_steps_counter = np.zeros(self.config['num_envs'])
        self.iter_bc_pretrain = 0.8e6 // (self.config['num_envs'] * self.num_steps)
        self.iter_bc_decay = 1.5 * self.iter_bc_pretrain

        self.bc_dagger = True
        self.bc_dagger_beta = 0.0

        self.curriculum_learning_iters = np.array([2e6 // self.nbatch, 4e6 // self.nbatch])
        self.curriculum_learning_start = 1

        self.hidden_state = self.net_model.init_hidden(self.config['num_envs'])
        self.val_env_obs = None

    def train(self):
        epinfobuf = deque(maxlen=10)
        t_trainstart = time.time()

        self.val_env_obs = self.val_env.reset()

        for iter in range(self.global_iter, self.max_iters):
            self.global_iter = iter
            t_iterstart = time.time()

            # Test and save model
            if iter % self.config['save_interval'] == 0 and logger.get_dir():
                with torch.no_grad():
                    res = self.rollout(val=True)
                    self.val_env_obs = self.val_env.reset()
                    obs, actions, returns, values, advs, log_probs, mpc_actions, epinfos = res
                logger.logkv("iter", iter)
                logger.logkv("test/total_timesteps", iter * self.nbatch)
                for key in epinfos.keys():
                    logger.logkv('test/' + key, epinfos[key])
                logger.dumpkvs()
                if epinfos['reward'] > self.best_avg_reward:
                    self.best_avg_reward = epinfos['reward']
                    is_best = True
                else:
                    is_best = False
                self.save_model(is_best=is_best, step=iter)

            # Run Rollout
            with torch.no_grad():
                obs, actions, returns, values, advs, log_probs, mpc_actions, epinfos = self.n_rollout(
                    repeat_num=self.config['train_rollout_repeat'])
            if epinfos:
                epinfobuf.extend(epinfos)
            lossvals = {'policy_loss': [],
                        'value_loss': [],
                        'policy_entropy': [],
                        'approxkl': [],
                        'clipfrac': [],
                        'bc_loss': []}

            # Train the Model
            bc_factor = - 1.0 / (self.iter_bc_decay - self.iter_bc_pretrain) * iter \
                        + self.iter_bc_decay / (self.iter_bc_decay - self.iter_bc_pretrain)
            bc_factor = min(1.0, max(0.0, bc_factor))
            opt_start_t = time.time()
            noptepochs = self.noptepochs
            for _ in range(noptepochs):
                # TODO add permutation of rollouts
                env_idc = torch.randperm(actions.shape[1])
                num_batches = int(np.ceil(actions.shape[1] / self.config['batch_size']))
                for x in range(num_batches):
                    b_start = x * self.config['batch_size']
                    b_end = min(b_start + self.config['batch_size'], actions.shape[1])

                    # Permute rollouts
                    # b_start = env_idc[int(b_start)]
                    # b_end = env_idc[int(b_end)]

                    large_maps, small_maps, states = obs
                    b_large_maps, b_small_maps, b_states = map(lambda p: p[:, b_start:b_end],
                                                               (large_maps, small_maps, states))
                    b_actions, b_returns, b_advs, b_log_probs, b_mpc_actions = map(lambda p: p[:, b_start:b_end],
                                                                                   (actions, returns, advs, log_probs,
                                                                                    mpc_actions))
                    hidden_state = self.net_model.init_hidden(batch_size=b_end - b_start)
                    for start in range(0, actions.shape[0], self.rnn_seq_len):
                        end = start + self.rnn_seq_len
                        slices = (arr[start: end] for arr in
                                  (b_large_maps, b_small_maps, b_states, b_actions,
                                   b_returns, b_advs, b_log_probs, b_mpc_actions))

                        # Model Update
                        info, hidden_state = self.net_train(*slices,
                                                            hidden_state=hidden_state, bc_factor=bc_factor)
                        # Log Losses
                        lossvals['policy_loss'].append(info['pg_loss'])
                        lossvals['value_loss'].append(info['vf_loss'])
                        lossvals['policy_entropy'].append(info['entropy'])
                        lossvals['approxkl'].append(info['approxkl'])
                        lossvals['clipfrac'].append(info['clipfrac'])
                        lossvals['bc_loss'].append(info['bc_loss'])

            tnow = time.time()
            int_t_per_epo = (tnow - opt_start_t) / float(self.noptepochs)
            print_cyan('Net training time per epoch: {0:.4f}s'.format(int_t_per_epo))
            fps = int(self.nbatch / (tnow - t_iterstart))
            if iter % self.config['log_interval'] == 0:
                logger.logkv("train/Learning rate", self.optimizer.param_groups[0]['lr'])
                logger.logkv("info/per_env_timesteps", iter * self.num_steps)
                logger.logkv("iter", iter)
                logger.logkv("info/total_timesteps", iter * self.nbatch)
                logger.logkv("info/fps", fps)
                logger.logkv('rollouts/returns', np.mean(returns[0, :]))
                for epinfo in epinfos:
                    for key in epinfo.keys():
                        logger.logkv('rollouts/' + key, epinfo[key])
                logger.logkv('info/time_elapsed', tnow - t_trainstart)
                for name, value in lossvals.items():
                    logger.logkv('train/' + name, np.mean(value))
                logger.logkv("train/bc_factor", bc_factor)
                logger.logkv("train/bc_dagger_beta", self.bc_dagger_beta)

                logger.dumpkvs()

    def test(self, render, val_id=0):
        self.net_model.eval()
        if self.env.num_envs != 1:
            raise ValueError('please use 1 env for testing and visualization')

        cum_rewards = []
        cum_seen_area = []
        cum_actions = []
        obs_uint8 = self.env.reset()
        if self.config['use_rgb_with_map']:
            print('RGB as input')
        hidden_state = self.net_model.init_hidden(batch_size=self.env.num_envs)
        for idx in range(self.num_steps):
            if render:
                self.env.render()

            obs = list(map(lambda p, device=self.device:
                           imagenet_rgb_preprocess(p, device), obs_uint8[:2]))
            # TODO states datatype
            # robot states
            obs.append(torch.from_numpy(obs_uint8[2]).float())

            res = self.net_model(*obs, hidden_state=hidden_state, deterministic=False)
            actions, log_probs, entropy, vals, hidden_state, act_logits = res

            cum_actions.append(actions.cpu().data.numpy()[0])
            obs_uint8, rewards, dones, infos = self.env.step(actions.cpu().data.numpy().flatten())
            cum_rewards.append(infos[0]['reward_so_far'])
            cum_seen_area.append(infos[0]['seen_area'])
            if idx == self.num_steps - 1:
                print('Seen area:', infos[0]['seen_area'])
                print('Total reward:', infos[0]['reward_so_far'])
                print('Collisions:', infos[0]['collisions'])
                print('Start pose:')
                start_pose = infos[0]['start_pose']
                print('x: {0:.3f}  z:{1:.3f}  yaw:{2:.3f}'.format(start_pose[0],
                                                                  start_pose[1],
                                                                  start_pose[2]))
        if render:
            while True:
                self.env.render()
        return cum_seen_area, cum_rewards, cum_actions

    def net_train(self, large_maps, small_maps, states, actions, returns, advs,
                  old_log_probs, mpc_actions, hidden_state, bc_factor=0.0):
        self.net_model.train()
        actions = torch.from_numpy(actions).float().to(self.device)
        returns = torch.from_numpy(returns).float().to(self.device)
        advs = torch.from_numpy(advs).float().to(self.device)
        old_log_probs = torch.from_numpy(old_log_probs).float().to(self.device)
        mpc_actions = torch.from_numpy(mpc_actions).float().to(self.device)

        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        large_maps = imagenet_rgb_preprocess(large_maps, device=self.device)
        small_maps = imagenet_rgb_preprocess(small_maps, device=self.device)
        states = states_preprocess(states, device=self.device)

        res = self.net_model(large_maps=large_maps,
                             small_maps=small_maps,
                             states=states,
                             hidden_state=hidden_state,
                             action=actions,
                             expert_action=mpc_actions)
        _, log_probs, entropy, vals_pred, hidden_state, act_logits, mpc_log_probs = res

        # Loss computation
        vals_pred = torch.squeeze(vals_pred)
        vf_loss = 0.5 * self.val_loss_criterion(vals_pred, returns)

        ratio = torch.exp(log_probs - old_log_probs)
        pg_loss1 = -advs * ratio
        pg_loss2 = -advs * torch.clamp(ratio,
                                       1 - self.config['cliprange'],
                                       1 + self.config['cliprange'])
        pg_loss = torch.mean(torch.max(pg_loss1, pg_loss2))

        bc_loss = -torch.mean(mpc_log_probs)

        policy_loss = (1 - bc_factor) * pg_loss + bc_factor * bc_loss

        loss = policy_loss - torch.mean(entropy) * self.config['ent_coef'] + \
               vf_loss * self.config['vf_coef']

        approxkl = 0.5 * torch.mean(torch.pow(old_log_probs - log_probs, 2))
        clipfrac = np.mean(np.abs(ratio.cpu().data.numpy() - 1.0) > self.config['cliprange'])

        self.optimizer.zero_grad()
        loss.backward()
        if self.config['max_grad_norm'] is not None:
            torch.nn.utils.clip_grad_norm_(self.net_model.parameters(),
                                           self.config['max_grad_norm'])
        self.optimizer.step()
        info = {
            'pg_loss': pg_loss.cpu().data.numpy(),
            'vf_loss': vf_loss.cpu().data.numpy(),
            'entropy': torch.mean(entropy).cpu().data.numpy(),
            'approxkl': approxkl.cpu().data.numpy(),
            'clipfrac': clipfrac,
            'bc_loss': bc_loss.cpu().data.numpy()
        }
        return info, hidden_state

    def n_rollout(self, repeat_num=1):
        # Run multiple rollouts and concatenate results

        n_large_maps, n_small_maps, n_states, nacts = [], [], [], []
        nret, nvals, nadvs = [], [], []

        nlog_probs, n_mpc_actions, nepinfos = [], [], []

        for j in range(repeat_num):
            res = self.rollout()
            obs, acts, ret, vals, advs, log_probs, mpc_actions, epinfos = res

            n_large_maps.append(obs[0])
            n_small_maps.append(obs[1])
            n_states.append(obs[2])
            nacts.append(acts)
            nret.append(ret)
            nvals.append(vals)
            nadvs.append(advs)
            nlog_probs.append(log_probs)
            n_mpc_actions.append(mpc_actions)
            nepinfos.append(epinfos)

        comb = map(lambda p: np.concatenate(p, axis=1),
                   (n_large_maps, n_small_maps, n_states, nacts, nret, nvals, nadvs, nlog_probs, n_mpc_actions))

        n_large_maps, n_small_maps, n_states, nacts, nret, nvals, nadvs, nlog_probs, n_mpc_actions = comb

        return (n_large_maps, n_small_maps, n_states), nacts, nret, nvals, nadvs, nlog_probs, n_mpc_actions, nepinfos

    def rollout(self, val=False):
        self.net_model.eval()
        mb_states, mb_large_maps, mb_small_maps, mb_rewards, mb_actions = [], [], [], [], []
        mb_values, mb_log_probs = [], []
        mb_mpc_actions = []

        # Set Curriculum Learning
        lvls_reached = np.where(self.global_iter >= self.curriculum_learning_iters)
        current_level = lvls_reached[0][-1] + 1 if lvls_reached[0].size > 0 else 0
        n_obstacles = current_level + 1

        if val:  # Test Rollout
            self.val_env.env_method('set_n_obstacles', n_obstacles)
            env = self.val_env
            # obs_uint8 = env.reset()
            obs_uint8 = self.val_env_obs
        else:  # Training Rollout
            # Set up Dagger and Expert in Env
            use_expert = self.global_iter <= self.iter_bc_pretrain
            comp_expert = self.global_iter <= self.iter_bc_decay
            if self.bc_dagger:
                self.bc_dagger_beta = np.clip((-1 / self.iter_bc_pretrain * self.global_iter + 1), 0.0, 1.0)
            self.env.env_method('set_use_expert_action', 1, use_expert, 'ig_greedy', self.bc_dagger,
                                self.bc_dagger_beta, comp_expert)

            self.env.env_method('set_n_obstacles', n_obstacles)
            env = self.env
            # Reset env
            obs_uint8 = env.reset()

        epinfos = {
            'reward': 0.0,
            'ig_reward': 0.0,
            'n_episodes': 0,
            'n_infeasible': 0,
            'n_timeout': 0,
            'n_collision': 0,
            'n_deadlocked': 0,
            'n_steps_avg': 0
        }
        # Init hidden states
        self.hidden_state = self.net_model.init_hidden(batch_size=self.config['num_envs'])

        # TODO move elsewhere
        state_keys = ['radius', 'heading_global_frame', 'angvel_global_frame', 'pos_global_frame', 'vel_global_frame']
        for idx in range(self.num_steps):

            # Preprocess Observations
            large_map = imagenet_rgb_preprocess(obs_uint8['ego_entropy_map'], device=self.device)
            small_map = imagenet_rgb_preprocess(obs_uint8['local_grid'], device=self.device)
            obs_states = np.stack([np.hstack([obs_uint8[key][i] for key in state_keys])
                                   for i in range(self.config['num_envs'])])
            states = states_preprocess(obs_states, device=self.device)

            # Save Observations
            mb_large_maps.append(obs_uint8['ego_entropy_map'])
            mb_small_maps.append(obs_uint8['local_grid'])
            mb_states.append(obs_states)

            # Query the Model
            res = self.net_model(large_maps=large_map,
                                 small_maps=small_map,
                                 states=states,
                                 hidden_state=self.hidden_state)
            actions, log_probs, entropy, vals, self.hidden_state, act_logits, expert_log_prob = res

            # Save Model Outputs
            mb_actions.append(np.squeeze(actions.cpu().data.numpy(), axis=0))
            mb_values.append(np.squeeze(vals.cpu().data.numpy(), axis=(0, -1)))
            mb_log_probs.append(np.squeeze(log_probs.cpu().data.numpy(), axis=0))

            # Step the Environment
            obs_uint8, rewards, dones, infos = env.step(actions.cpu().data.numpy().flatten())

            # Save Rewards Expert Actions
            mb_rewards.append(rewards)
            mb_mpc_actions.append(np.stack([info["mpc_actions"] for info in infos]))

            # TODO Reset hidden states for completed episodes
            done_idc = np.array(dones.nonzero()).squeeze()
            # self.hidden_state[:, done_idc, :] *= 0.0

            # Save Episode Infos
            self.reward_counter += rewards
            self.reward_ig_counter += np.asarray([info["ig_reward"] for info in infos])
            # if done_idc.size > 0 or idx == self.num_steps-1:
            #     for i in done_idc.tolist():
            if idx == self.num_steps - 1:
                for i in range(self.config['num_envs']):
                    epinfos['n_episodes'] += 1
                    epinfos['reward'] += self.reward_counter[i]
                    self.reward_counter[i] = 0.0
                    epinfos['ig_reward'] += self.reward_ig_counter[i]
                    self.reward_ig_counter[i] = 0.0
                    epinfos['n_steps_avg'] += infos[i]['step_num']
                    epinfos['n_timeout'] += infos[i]["ran_out_of_time"]
                    epinfos['n_collision'] += infos[i]["in_collision"]
                    epinfos['n_deadlocked'] += infos[i]["deadlocked"]
                    epinfos['n_infeasible'] += infos[i]["is_infeasible"]

        mb_large_maps = np.stack(mb_large_maps, axis=0)
        mb_small_maps = np.stack(mb_small_maps, axis=0)
        mb_states = np.stack(mb_states, axis=0)
        mb_rewards = np.stack(mb_rewards, axis=0)
        mb_actions = np.stack(mb_actions, axis=0)
        mb_mpc_actions = np.stack(mb_mpc_actions, axis=0)
        mb_values = np.stack(mb_values, axis=0)
        mb_log_probs = np.stack(mb_log_probs, axis=0)

        if epinfos['n_episodes'] > 0:
            for key in epinfos.keys():
                if key == "n_episodes":
                    continue
                epinfos[key] = epinfos[key] / epinfos['n_episodes']

        # Compute Value of last state for advantage computation
        # - Preprocess Observations
        large_map = imagenet_rgb_preprocess(obs_uint8['ego_entropy_map'], device=self.device)
        small_map = imagenet_rgb_preprocess(obs_uint8['local_grid'], device=self.device)
        obs_states = np.stack([np.hstack([obs_uint8[key][i] for key in state_keys])
                               for i in range(self.config['num_envs'])])
        states = states_preprocess(obs_states, device=self.device)
        # - Query the model
        res = self.net_model(large_maps=large_map,
                             small_maps=small_map,
                             states=states,
                             hidden_state=self.hidden_state)
        actions, log_probs, entropy, vals, hidden_state, act_logits, expert_log_prob = res

        last_values = np.squeeze(vals.cpu().data.numpy(), axis=-1)

        # Compute Advantages
        mb_advs, mb_returns = self.get_gae(mb_rewards, mb_values, last_values)

        return (mb_large_maps, mb_small_maps, mb_states), mb_actions, mb_returns, mb_values, mb_advs, \
               mb_log_probs, mb_mpc_actions, epinfos

    def get_gae(self, rewards, value_estimates, value_next):
        mb_advs = np.zeros_like(rewards)
        lastgaelam = 0
        value_estimates = np.concatenate((value_estimates, value_next), axis=0)
        for t in reversed(range(rewards.shape[0])):
            delta = rewards[t] + self.config['gamma'] * value_estimates[t + 1] - value_estimates[t]
            mb_advs[t] = lastgaelam = delta + self.config['gamma'] * self.config['lam'] * lastgaelam
        mb_returns = mb_advs + value_estimates[:-1, ...]
        return mb_advs, mb_returns
