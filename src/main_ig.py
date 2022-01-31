import random
from datetime import datetime
import os
import numpy as np
import torch

from algos.ppo_ig import PPOAgentIG
from configs.parse_args import parse_arguments
from utils.color_print import *
from utils.subproc_vec_env import SubprocVecEnv

from gym_collision_avoidance.envs.collision_avoidance_env import CollisionAvoidanceEnv
import gym


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def create_env(n_envs=1, subproc=True):
    def make_env():
        def _thunk():
            env = CollisionAvoidanceEnv()
            return env

        return _thunk

    if subproc:
        return SubprocVecEnv([make_env()
                              for i in range(n_envs)])


def main():
    args = parse_arguments()
    print_green('Program starts at: \033[92m %s '
                '\033[0m' % datetime.now().strftime("%Y-%m-%d %H:%M"))
    args.device = None
    args.device = torch.device("cuda:0" if torch.cuda.is_available()
                                           and not args.disable_cuda
                               else "cpu")
    set_random_seed(args.seed)

    env = create_env(args.num_envs, subproc=True)
    if args.test:
        val_env = None
    else:
        val_env = create_env(min(args.num_envs, 24), subproc=True)

    def get_latest_run_id(log_path):
        p = os.listdir(log_path)
        p = [item for item in p if os.path.isdir(log_path + '/' + item)]

        if len(p) > 0:
            p = list(map(lambda fname: int(fname.split('_')[1]), p))
            p.sort()
            id = p[-1]
        else:
            id = 0

        return id

    log_path = os.getcwd() + '/data'
    save_path = os.path.join(log_path, "log_{}".format(get_latest_run_id(log_path) + 1))
    args.save_dir = save_path

    print("Log path: {}".format(save_path))

    # Save plot trajectories
    for i in range(args.num_envs):
        plot_save_dir = save_path + '/figures_train/figs_env' + str(i) + '/'
        # os.makedirs(plot_save_dir, exist_ok=True)
        env.env_method('set_plot_save_dir', plot_save_dir, indices=i)
        env.env_method('set_n_env', args.num_envs, i, False, indices=i)

        plot_save_dir = save_path + '/figures_test/figs_env' + str(i) + '/'
        val_env.env_method('set_plot_save_dir', plot_save_dir, indices=i)
        val_env.env_method('set_n_env', args.num_envs, i, True, indices=i)
        val_env.env_method('set_use_expert_action', 0, False, None, False, 1.0, False, indices=i)
        if i != 0:
            env.env_method('set_plot_env', False, indices=i)
            # val_env.env_method('set_plot_env', False, indices=i)

    args.rnn_type = None
    args.fix_cnn = True
    agent = PPOAgentIG(env=env, args=args, val_env=val_env)

    if args.test:
        for i in range(0, 100):
            agent.test(render=args.render, val_id=i)
    else:
        try:
            agent.train()
        except KeyboardInterrupt:
            pass
        print('Saving model to {}'.format(args.save_dir))
        agent.save_model(False)

    # env.close()


if __name__ == "__main__":
    main()
