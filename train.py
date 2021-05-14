from runner import *
from opt import *
from model import *
from cartpole import *
from invertedDoublePendulum import *
from bipedEnv import *

'''
use this script to train your controller
'''

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, help = "which environment to use: cartpole, invertedDoublePendulum and bipedal")
    parser.add_argument('--opt', type=str, help = "which DRL algorithm to use: reinforce, actor-critic or ppo")
    parser.add_argument('--save_path', type=str, help = "where to save the model")
    args = parser.parse_args()
    env_name = args.env
    opt_name = args.opt
    save_path = args.save_path

    if(env_name == "cartpole"):
        env = CartPoleBulletEnv(False)
        actor = Actor(env.observation_space.shape[0], env.action_space.shape[0], env.action_space, hidden=[128, 64])
        critic = Critic(env.observation_space.shape[0], 0, 1/(1-0.99), hidden =[128, 64])
        curriculum = False
        batch_size = 1024
        max_steps = 1000
    elif(env_name == "invertedDoublePendulum"):
        env = InvertedDoublePendulumBulletEnv(False)
        actor = Actor(env.observation_space.shape[0], env.action_space.shape[0], env.action_space, hidden=[128, 64])
        critic = Critic(env.observation_space.shape[0], 0, 10/(1-0.99), hidden =[128, 64])
        curriculum = False
        batch_size = 1024
        max_steps = 1000
    else:
        env = bipedEnv(False)
        actor = Actor(env.observation_space.shape[0], env.action_space.shape[0], env.action_space, hidden=[128, 64])
        critic = Critic(env.observation_space.shape[0], 0, 4/(1-0.99), hidden =[128, 64])
        curriculum = True
        env.setKpandKd(1000)
        batch_size = 2048
        max_steps = 50

    env.reset()
    s_norm = Normalizer(env.observation_space.shape[0])
    if(opt_name == "reinforce"):
        reinforce(save_path, env, actor, critic, s_norm, 500, batch_size,  batch_size, 3e-4, 0.5, max_steps, curriculum)
    elif(opt_name == "actor-critic"):
        actor_critic(save_path, env, actor, critic, s_norm, 500, batch_size,  batch_size, 3e-4, 3e-4, 0.5, max_steps, curriculum)
    else:
        PPO(save_path, env, actor, critic, s_norm, 0.2, 500, batch_size,  64, 3e-4, 3e-4, 0.5, max_steps, curriculum)
