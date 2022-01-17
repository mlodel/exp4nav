import random
from datetime import datetime

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
        val_env = create_env(min(args.num_envs, 8), subproc=True)

    agent = PPOAgentIG(env=env, args=args, val_env=val_env)

    if args.test:
        for i in range(0, 100):
            agent.test(render=args.render, val_id=i)
    else:
        agent.train()

    # TODO SAVE WHEN CTRL+C
    # try:
    #     #model = ALGOS[args.algo].load(model_path, env=env)
    #     model.learn(n_timesteps, **kwargs)
    # except KeyboardInterrupt:
    #     pass
    #
    #
    # # Only save worker of rank 0 when using mpi
    # if rank == 0:
    #     print("Saving to {}".format(save_path))
    #
    #     model.save("{}/{}".format(save_path, env_id))

    env.close()


if __name__ == "__main__":
    main()
