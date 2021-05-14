import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pybullet as p
import time
import math
import numpy as np
from IPython import embed

class pdController(object):
    '''
    This is the class of stable pd controller
    '''
    def __init__(self, bipedId, clientId, timeStep):

        self.bipedId = bipedId # character id
        self.clientId = clientId # simulation engine id
        self.kp = [0, 0, 0, 500, 500, 500, 500, 200, 200] # kp parameters for controller  in different dof
        self.kd = [0, 0, 0, 50, 50, 50, 50, 20, 20] #kd parameter for controller in different dof
        self.kp = np.diagflat(self.kp)
        self.kd = np.diagflat(self.kd)
        self.init_kp = self.kp.copy() # save the initial kp and kd values for curriculum learning
        self.init_kd = self.kd.copy()
        self.timeStep = timeStep #simulation time step
        self.num_joints = p.getNumJoints(self.bipedId, physicsClientId=self.clientId) # get the number of dof in character

    def getPosandVel(self):
        '''
        return two numpy arrays
        one for position of different dof
        one for velocity of different dof
        '''

        pos=[]
        vel=[]
        for i in range(self.num_joints):
            state = p.getJointState(self.bipedId, i, physicsClientId=self.clientId)
            pos.append(state[0])
            vel.append(state[1])
        
        return np.array(pos), np.array(vel)

    def getForce(self, targetPos, vel):
        '''
        return computed force for each dof by stable pd controller,
        you need to write part of this function, please refer to "Stabel Propotional-Derivative Controllers"'s
        section 3.1 and 3.3 for details

        targetPos : a numpy array of target position(angle) for each dof
        vel : target root velocity in facing direction
        return : a numpy array of computed torques 
        '''
        pos_now, vel_now = self.getPosandVel()
        targetPos = np.array([0]*3 + targetPos.tolist())
        targetPos[0] = pos_now[0]
        vel = np.array([vel, 0, 0, 0, 0, 0, 0, 0, 0]) # all the dof except for the face direction 's target velocity is 0
        pos_part = self.kp.dot(pos_now - targetPos) #position part in pd controller
        vel_part = self.kd.dot(vel_now) #velocity part in pd controller
        M = p.calculateMassMatrix(self.bipedId, pos_now.tolist()) #mass matrix
        M = np.array(M)
        # add your code here, please compute the qddot part
        # hint: to compute the external force and centrifugal force, you can use p.calculateInverseDynamics()
        M = (M + self.kd * self.timeStep)
        c = p.calculateInverseDynamics(self.bipedId, pos_now.tolist(), vel_now.tolist(), [0]*9)
        c = np.array(c)
        b = -pos_part - vel_part -self.kp.dot(vel_now)*self.timeStep - c
        qddot = np.linalg.solve(M, b) 
        tau = -pos_part - vel_part - self.kd.dot(qddot) * self.timeStep-self.kp.dot(vel_now)*self.timeStep + self.kd.dot(vel)

        return tau


class bipedEnv(gym.Env):

    '''
    the class of bipedal character
    '''
    metadata = {'render.modes' : ['human']}


    def __init__(self, renders = True):

        self.renders = renders #set renders to True to render the biped environment, in training please set it to False
        if(self.renders == True):
            self.clientId = p.connect(p.GUI)
        else:
            self.clientId = p.connect(p.DIRECT)

        #simulation parameters
        self.timeStep = 0.002  # simulation step
        self.num_substep = 30 # how many simulation steps to simulate in one call of step()
        self.num_step = 0 # how many times you call step()
        self.episode_length = 100 # how many times you can call step()
        self.configure() #set simulation engine parameters
        self.pdCon = pdController(self.bipedId, self.clientId, self.timeStep) #spd controller
        #current setting explanation: 
        #Every simulation step is 0.002s, in one call of step(), you would forward the simulation 30 steps, which means 0.06s
        #you can call step() 50 times, which means 3s, so the task is to train a character learn to walk for 3s, the episode_length can be adjusted

        #gym settings
        self.action_space = spaces.Box(low = -1, high = 1, shape=(6, )) #action space, we only control 6 revolute joints
        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape = (21, )) # observation space, include character's information
        self.scale_action = np.array([math.pi/2, math.pi/2, math.pi/2, math.pi/2
        , 0.5, 0.5]) # scale the action space        
        self.vel =0 #current root target velocity
        self.vel_target = 1.0 #root target velocity
        self.pos_current = 0 # current root position
        self.pos_prev = 0 #root position in last frame
        self.obv = None
   

        #reward weights
        self.w_action = 1
        self.w_velocity = 3
        self.w_live = 1
        self.live_bonus = 4
        self.w_upright = 1
       
        #Symmetry Matrix
        # add your code here, you need to write two symmetry matrix, one for action and  another for observation
        self.M_action = np.zeros((6, 6))
        self.M_state = np.zeros((21, 21))

        self.M_action=np.array(
        [[0, 1, 0, 0, 0, 0], 
        [1, 0, 0, 0, 0, 0], 
        [0, 0, 0, 1, 0, 0], 
        [0, 0, 1, 0, 0, 0], 
        [0, 0, 0, 0, 0, 1], 
        [0, 0, 0, 0, 1, 0]])

        self.M_state = np.zeros((21, 21))

        #init M_state
        self.M_state[0,0] = 1
        self.M_state[1:7, 1:7] = self.M_action
        self.M_state[7, 7] = 1
        self.M_state[8:14, 8:14] = self.M_action
        self.M_state[14, 14] = 1
        self.M_state[15:19, 15:19] = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0]]
        )
        self.M_state[19, 19] = 1
        self.M_state[20, 20] = 1
       

        # self.M_action = None
        # self.M_state = None



        #curriculum setting
        self.kpandkd_start = self.pdCon.init_kp[2,2] # kp and kd for assistance force at the beginning
        self.kpandkd_end = self.kpandkd_start*0.75*0.75 # kp and kd for assistance force at last

    def step(self, action):
        '''
        step function
        input: action,  a numpy array with shape (6, )
        return: obv : character 's state after the action
               rwd : reward
               done : whether the character fall to plane or task is over
               info : some information for debug
        '''
        
        
        # compute current kp and kd for assistance force in root joint
        kpandkd = self.kpandkd_start - (self.num_step)/(self.episode_length)*(self.kpandkd_start - self.kpandkd_end)
        # set kp and kd to root joint
        self.pdCon.kp[2, 2] =kpandkd
        self.pdCon.kd[2, 2] =0.1*kpandkd
        self.pdCon.kd[0, 0] =kpandkd

        # compute current root target vecloity
        self.vel = np.min([2 * self.num_step * self.timeStep * self.num_substep, self.vel_target])
        
        self.pos_prev =  p.getLinkState(self.bipedId, 2,  physicsClientId=self.clientId)[0][1]


        action_real = action * self.scale_action # get target position for each joint      
        joint_idx = np.arange(self.pdCon.num_joints).tolist()
        forces_array = []
        for i in range(self.num_substep):
            forces = self.pdCon.getForce(action_real, self.vel).tolist() # compute force
            forces_array.append(forces)
            p.setJointMotorControlArray(self.bipedId, joint_idx, p.TORQUE_CONTROL, forces= forces) # apply force to each dof
            p.stepSimulation(physicsClientId=self.clientId) # simulation
            if(self.renders==True): # if render is set to True, render the scene
                pos_root = p.getLinkState(self.bipedId, 2,  physicsClientId=self.clientId)[0]
                p.resetDebugVisualizerCamera(2, 90, 0, [1, pos_root[1], 1 ], self.clientId)
                time.sleep(self.timeStep)
        self.num_step += 1 # current step

        self.pos_current =  p.getLinkState(self.bipedId, 2,  physicsClientId=self.clientId)[0][1] # current root postition
        obv = self.getObv() # get observation state
        self.obv =obv
        forces_array = np.array(forces_array).mean(0)
        rwd_action, rwd_live, rwd_upright, rwd_vel = self.getRwd(action_real - self.obv[1:7], obv)
        done = self.getDone(obv) # get done flag, check whether the task is over
        
        rwd = self.w_action*rwd_action + self.w_live* rwd_live + self.w_upright*rwd_upright + self.w_velocity*rwd_vel # compute reward
        # record debug information
        info={}
        info["rwd_action"] = rwd_action
        info["rwd_live"] = rwd_live
        info["rwd_upright"] = rwd_upright
        info["rwd_vel"] = rwd_vel
        
        return obv, rwd, done, info

    def setKpandKd(self,kd):
        '''
        set kp and kd for assistance force at the beginning
        '''
        self.pdCon.init_kp[2,2] = kd
        self.pdCon.init_kd[0,0] = kd
        self.pdCon.init_kd[2,2] = 0.1*kd

        self.kpandkd_start = self.pdCon.init_kp[2,2]
        self.kpandkd_end = self.kpandkd_start*0.75*0.75


    def reset(self):
        '''
        when the task is over, reset the state of bipedal character and restart the simulation

        return: character's state after reset
        '''
        p.resetSimulation()
        self.configure()
        #reset the biped to initial pose and velocity
        start_pose=[0, 0, 0., 0.05, 0, 0, 0, 0, 0]
        for i in range(p.getNumJoints(self.bipedId, physicsClientId=self.clientId)):
            p.resetJointState(self.bipedId, i, start_pose[i], 0, physicsClientId= self.clientId)
        self.num_step = 0
        self.pdCon.kp = self.pdCon.init_kp.copy()
        self.pdCon_kd = self.pdCon.init_kd.copy()
        self.vel = 0
        self.obv= self.getObv()
        return self.obv

    def getObv(self):
        '''
        return current state, please notice the states' content
        return: bipedal character's current state
        '''
        obv=[]
        pos, vel =self.pdCon.getPosandVel()
        obv += pos[-7:].tolist() #  root 's angle  in XY plane and position of controlled 6 joints 
        obv += vel[-7:].tolist() #  root 's anbular velocty in XY plane and velocity of controlled 6 joints
        pos_root = np.array(p.getLinkState(self.bipedId, 2, physicsClientId=self.clientId)[0]) # root position
        pos_leftfoot = np.array(p.getLinkState(self.bipedId, 8, physicsClientId=self.clientId)[0]) - pos_root #left foot's relative postion to root
        pos_rightfoot = np.array(p.getLinkState(self.bipedId, 7, physicsClientId=self.clientId)[0]) - pos_root # right foot's relative position to root 
        obv.append(pos_root[2]) # root's height
        obv += [pos_leftfoot[1], pos_leftfoot[2]] # left end effector
        obv += [pos_rightfoot[1], pos_rightfoot[2]] # right end effector
        root_vel = p.getLinkState(self.bipedId, 2, computeLinkVelocity=1, physicsClientId=self.clientId)[6][1] # root velocity in face direction
        obv += [root_vel] # root velocity
        obv += [self.vel]
        obv = np.array(obv)
        return obv

    def getRwd(self, forces, obv):
        '''
        compute reward
        return: reward for different aims
        '''
        vel = (self.pos_current - self.pos_prev)/(self.num_substep*self.timeStep)
        rwd_vel = -abs(vel - self.vel)
        rwd_action = -np.linalg.norm(forces)
        rwd_upright = -abs(obv[0])
        
        return rwd_action, self.live_bonus, rwd_upright, rwd_vel 

    def getDone(self, obv):
        '''
        check whether the character fails the task
        1. The task is over
        2. The root is about to lose the balance
        3. numberical error
        4. The y position of root joint is too low

        return bool varibale: True for over
        '''
    
        return not((self.num_step<self.episode_length) and  (np.isfinite(obv).all())
        and (abs(obv[14]-1.09)<0.2))

    def getSymmetryState(self, state):
        '''
        return symmetry states
        '''

        return self.M_state.dot(state)

    def getSymmetryAction(self, action):
        '''
        return symmetry actions
        '''

        return self.M_action.dot(action)

    def setRenders(self, renders):
        self.renders = renders
    

    def render(self, mode='human'):
        pass


    def close(self):
        pass


    def configure(self):
        '''
        set parameters of simulation engines
        '''
        p.setTimeStep(self.timeStep)
        p.setGravity(0, 0, -10)
        self.bipedId = p.loadURDF("./human_model/biped2d_pybullet.urdf", basePosition=[0, 0, 1.095], useFixedBase=True, flags=p.URDF_MAINTAIN_LINK_ORDER )
        self.planeId = p.loadURDF("./human_model/plane.urdf", globalScaling = 2)

        #clear the torque in every joint
        for i in range(p.getNumJoints(self.bipedId, physicsClientId=self.clientId)):
            p.setJointMotorControl2(self.bipedId, i, p.POSITION_CONTROL, force=0, physicsClientId=self.clientId)

        



if __name__ == "__main__":
    env=bipedEnv()
    env.reset()
    action_list=None
    obv_list=None

    
    while(1):
        action=[]
        for i in range(6):
            action.append(0)
        action=np.array(action)       
        obv, rwd, done, info =env.step(action)
        if(done == True):
            env.reset()
         
 
       

