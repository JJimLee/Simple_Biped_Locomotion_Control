import numpy as np
import torch
from IPython import embed
class Runner(object):
    '''
    the class of runner, which would collect sample data
    '''
    def __init__(self, env, s_norm, actor, critic, sample_size, gamma, lam, max_steps=50):
        self.env = env # simulation environment
        self.s_norm = s_norm # state normalization module
        self.actor = actor # actor neural network
        self.critic = critic # critic neural network
        self.obs = np.asarray(env.reset().reshape((1, -1)), dtype=np.float32) #initialize observation
        self.obs[:] = env.reset()
        self.toTorch = lambda x: torch.Tensor(x).float() #convert data to torch tensor
        self.sample_size = sample_size # the number of samples we need
        self.dones = None # dones bool varibale 

        self.lam = lam  # lambda used in GAE
        self.gamma = gamma  # discount rate
        self.v_min = self.critic.v_min
        self.v_max = self.critic.v_max

        #parameters for curriculum learning
        self.max_steps = max_steps
        self.current_step = 0
        

    def run(self):
        '''
        return collected lsit of samples
        '''
        self.current_step = 0
        n_steps = self.sample_size
        self.s_norm.update() #update state normlaization module
        self.obs[:] = self.env.reset() # reset environment states
        mb_obs=[] #observation list
        mb_acs=[] # actions list
        mb_rwds=[] # rewaid list
        mb_vpreds=[] #predicted values list
        mb_alogps=[] #log probability of action list
        mb_obs_next=[] # next observation list
        mb_dones=[] # dones list
        mb_vpreds_next=[] # next predicted value list
        mb_fails = []
      
        for _ in range(n_steps):
            obst = self.toTorch(self.obs)
            self.s_norm.record(obst)
            obst_norm = self.s_norm(obst) #normalize observation
            
            with torch.no_grad():
                m = self.actor.act_distribution(obst_norm)
                acs = m.sample() # sample action from gaussian distrbution
                alogps = torch.sum(m.log_prob(acs), dim=1).numpy()
                acs = acs.view(-1).numpy()
                vpreds = self.critic(obst_norm).view(-1).numpy() # compute predicted state values
               
          
            mb_obs.append(self.obs.copy().reshape(-1))
            mb_acs.append(acs.reshape(-1))
            mb_vpreds.append(vpreds)
            mb_alogps.append(alogps)


            self.obs[:], rwds, self.dones, infos = self.env.step(acs) # step action
            self.current_step+=1
            if(self.dones==False):
                    mb_fails.append(False)
            else:
                    mb_fails.append(True)
            if(self.current_step > self.max_steps):
                self.dones = True
           

            mb_dones.append(self.dones)
            mb_obs_next.append(self.obs.copy().reshape(-1))
            #compute next states's value function
            if(self.dones == True):
                if(mb_fails[-1]==False):
                     with torch.no_grad():
                        obst = self.toTorch(self.obs)
                        self.s_norm.record(obst)
                        obst_norm = self.s_norm(obst)
                        vpreds_next = self.critic(obst_norm).view(-1).numpy()
                        #print("vpreds_next:{}".format(vpreds_next.shape))
                        mb_vpreds_next.append(vpreds_next.reshape(-1))
                else:
                    mb_vpreds_next.append(np.asarray([0], dtype=np.float32))
                self.obs[:] = self.env.reset()
                self.current_step = 0
            else:
                with torch.no_grad():
                        obst = self.toTorch(self.obs)
                        self.s_norm.record(obst)
                        obst_norm = self.s_norm(obst)
                        vpreds_next = self.critic(obst_norm).view(-1).numpy()
                        #print("vpreds_next:{}".format(vpreds_next.shape))
                        mb_vpreds_next.append(vpreds_next.reshape(-1))
            mb_rwds.append([rwds])
           


        mb_acs = np.asarray(mb_acs, dtype = np.float32)
        mb_obs = np.asarray(mb_obs, dtype = np.float32)
        mb_dones = np.asarray(mb_dones, dtype = np.bool)
        mb_obs_next = np.asarray(mb_obs_next, dtype = np.float32)
        mb_rwds = np.asarray(mb_rwds, dtype = np.float32)
        mb_vpreds = np.asarray(mb_vpreds, dtype = np.float32)
        mb_vpreds_next = np.asarray(mb_vpreds_next, dtype = np.float32)
        mb_alogps = np.asarray(mb_alogps, dtype = np.float32)

        #advantage functions and taregt value functions
        mb_advs = np.zeros_like(mb_rwds)
        mb_vtars = np.zeros_like(mb_rwds)
        mb_acc_rwds = np.zeros_like(mb_rwds)
        #delta in GAE algorithm
        mb_delta = mb_rwds + self.gamma * mb_vpreds_next - mb_vpreds
     
        #compute the GAE advantage and value function
        #add your code here
        
        lastgaelam = 0
        lastvtar=0
        for t in reversed(range(n_steps)):
            #mb_advs[t] = None
            #mb_vtars[t] = None
            coe = 1-mb_dones[t] 
            lastgaelam =mb_delta[t] + coe*self.gamma*lastgaelam*self.lam
            mb_advs[t] = lastgaelam
            lastvtar=mb_advs[t]+mb_vpreds[t]
            mb_vtars[t] = lastvtar
        



        #compute the accumulated reward
        #add your code here
        last_rwd = 0
        for t in reversed(range(n_steps)):
            mb_acc_rwds[t] = mb_vtars[t]*mb_advs[t]
            
        #return collected samples
        rollouts={}
        rollouts["acs"] = mb_acs
        rollouts["obs"] = mb_obs
        rollouts["obs_next"] = mb_obs_next
        rollouts["rwds"] = mb_rwds
        rollouts["advs"] = mb_advs
        rollouts["vtars"] = mb_vtars
        rollouts["alogps"] = mb_alogps
        rollouts["dones"] = mb_dones
        rollouts["vpreds"] = mb_vpreds
        rollouts["acc_rwds"] = mb_acc_rwds

        return rollouts

#objective = torch.mean(acc_rwds_batch*alogps_batch)
    def testModel(self, num_epoch =10,render=False):
        '''
        test model
        '''
 
        with torch.no_grad():
            rwd_acc=0
            test_step = 0
            for i in range(num_epoch):
                self.current_step = 0
                obs = self.env.reset()
                obs= torch.Tensor(obs).float()#.view(-1,1)
                done = False
                obs_norm = self.s_norm(obs)
                while(not done):
                    ac = self.actor(obs_norm).numpy()
                    ac = ac.reshape(-1)
                    obs, rwd, done , info = self.env.step(ac)
                
                    if(render==True):
                        self.env.render()
                    rwd_acc += rwd
                
                    test_step+=1
                    obs = torch.Tensor(obs).float()#.view(-1,1)
                    obs_norm = self.s_norm(obs)
                    self.current_step+=1
                    if(self.current_step>=self.max_steps):
                        break


            print("avg_rwd:{}".format(rwd_acc/num_epoch))
            print("avg_steps:{}".format(test_step/num_epoch))
            return rwd_acc/num_epoch, test_step/num_epoch

            
def evaluate(env, actor, s_norm, num_epoch = 10):
        with torch.no_grad():
            rwd_acc=0
            test_step=0
            for i in range(num_epoch):
                obs = env.reset()
                env.render()
                
                obs= torch.Tensor(obs).float()#.view(-1,1)
                
             
                done = False
                obs_norm = s_norm(obs)
                while(not done):
                    ac = actor(obs_norm).numpy()
                    ac = ac.reshape(-1)
                    obs, rwd, done , info = env.step(ac)  
                    env.render()  
                   
                    rwd_acc += rwd
                    test_step+=1
                    obs = torch.Tensor(obs).float()#.view(-1,1)
                    obs_norm = s_norm(obs)
            print("rwd_acc:{}".format(rwd_acc/num_epoch))
            print("test_step:{}".format(test_step/num_epoch))
          
            return rwd_acc/num_epoch, test_step/num_epoch

        


if __name__ == "__main__":
    #import pybullet_envs
    import gym
    from model import *
    from gym_biped.envs.bipedEnv import *
    #env = gym.make("Walker2d-v2")
    #env.isRender = False
    env = bipedEnv(True)
    env.reset()
   
    s_norm, actor, critic = load_model("./Walker2d-Bullet-curriculum-phase-1-50-1.0/checkpoint_1990.tar")
    runner =  Runner(env, s_norm, actor, critic, 4096, 0.99, 0.95, 1)
    runner.env.setKpandKd(0)
    runner.testModel(1)


      

      