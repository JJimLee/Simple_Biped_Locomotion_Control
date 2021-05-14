import os, inspect



import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
import pybullet_data
from IPython import embed




class InvertedDoublePendulumBulletEnv(gym.Env):
  metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

  def __init__(self, renders=True):
    # start the bullet physics server
    self._renders = renders
    if (renders):
      p.connect(p.GUI)
    else:
      p.connect(p.DIRECT)
    self.theta_threshold_radians = 12 * 2 * math.pi / 360
    self.x_threshold = 2.4  #2.4
    high = np.array([
        self.theta_threshold_radians * 2,
        np.finfo(np.float32).max,
        self.x_threshold * 2,
        np.finfo(np.float32).max,
        self.theta_threshold_radians * 2,
        np.finfo(np.float32).max
    ])



    self.action_space = spaces.Box(-np.array([1]), np.array([1]), dtype = np.float32)
    self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    self.seed()
    #    self.reset()
    self.viewer = None
    self._configure()
    self.reset()

  def _configure(self, display=None):
    self.display = display

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def step(self, action):
    force = action*200

    p.setJointMotorControl2(self.cartpole, 0, p.TORQUE_CONTROL, force=force)
    p.stepSimulation()

    self.state = p.getJointState(self.cartpole, 1)[0:2] + p.getJointState(self.cartpole, 0)[0:2] + p.getJointState(self.cartpole, 2)[0:2]
    theta, theta_dot, x, x_dot, theta_upper, theta_dot_upper = self.state
    y = np.cos(theta) + np.cos(theta_upper)
    done =  y<1 or abs(x)>2
    if(self._renders == True):
        time.sleep(self.timeStep)

    reward = 10 - 1e-3*theta_dot**2 - 5e-3*theta_dot_upper**2 - 0.01*x**2 - (y-2)**2
    
    return np.array(self.state), reward, done, {}

  def reset(self):
    #    print("-----------reset simulation---------------")
    p.resetSimulation()
    self.cartpole = p.loadURDF("./human_model/inverteddoublependulum.urdf",
                               [0, 0, 0])
    
    p.changeDynamics(self.cartpole, -1, linearDamping=0, angularDamping=0)
    p.changeDynamics(self.cartpole, 0, linearDamping=0, angularDamping=0)
    p.changeDynamics(self.cartpole, 1, linearDamping=0, angularDamping=0)
    p.changeDynamics(self.cartpole, 2, linearDamping=0, angularDamping=0)
    self.timeStep = 0.02
    p.setJointMotorControl2(self.cartpole, 1, p.VELOCITY_CONTROL, force=0)
    p.setJointMotorControl2(self.cartpole, 0, p.VELOCITY_CONTROL, force=0)
    p.setJointMotorControl2(self.cartpole, 2, p.VELOCITY_CONTROL, force=0)
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(self.timeStep)
    p.setRealTimeSimulation(0)

    randstate = self.np_random.uniform(low=-0.01, high=0.01, size=(4,))
    p.resetJointState(self.cartpole, 1, randstate[0], randstate[1])
    p.resetJointState(self.cartpole, 0, randstate[2], randstate[3])
    #print("randstate=",randstate)
    self.state = p.getJointState(self.cartpole, 1)[0:2] + p.getJointState(self.cartpole, 0)[0:2] + p.getJointState(self.cartpole, 2)[0:2]
    #print("self.state=", self.state)
    return np.array(self.state)

  def render(self, mode='human', close=False):
    return


if __name__ =="__main__":
    env = InvertedDoublePendulumBulletEnv(True)
    print(env.action_space)
    print(env.observation_space)
    while(1):
        action = env.action_space.sample()
        obv, rwd, done, info =env.step(0)
        if(done):
            env.reset()        
