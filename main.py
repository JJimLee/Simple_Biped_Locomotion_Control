from runner import *
from bipedEnv import *
from cartpole import *
from invertedDoublePendulum import *
from model import *
'''
use this script to test trained model
'''
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, help = "which environment to use: cartpole, invertedDoublePendulum and bipedal")
    parser.add_argument('--load_path', type=str, help = "where to load the model")
    args = parser.parse_args()
    env_name = args.env
    load_path = args.load_path
    if(env_name == "cartpole"):
        env = CartPoleBulletEnv(True)
    elif(env_name == "invertedDoublePendulum"):
        env = InvertedDoublePendulumBulletEnv(True)
    else:
        env = bipedEnv(True)
        env.setKpandKd(0)
    env.reset()
    s_norm, actor, critic = load_model(load_path)
    evaluate(env,actor, s_norm)