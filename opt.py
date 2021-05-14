import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from runner import *
import os



def adjust_KpandKd(env, alpha):
    '''
    adjust environment's kp and kd for assistance torque on the root joint
    env: simluation environment
    alpha : shrinking factor
    '''
    kp = env.pdCon.init_kp.copy()
    env.setKpandKd(kp[2, 2]*alpha)


def PPO(save_path,
        env,
        actor,
        critic,
        s_norm,
        clip_param,
        ppo_epoch,
        batchsize,
        num_mini_batch,
        lr_actor=None,
        lr_critic=None,
        max_grad_norm=None,
        env_maxsteps = 500,
        curriculum = True

        ):
        '''
        PPO optimzation algorithm
        env: simulation environment
        actor: actor neural network
        critic: critic network
        s_norm : state normalization module
        clip_param : clip parameters in PPO
        ppo_epoch : epochs of PPO optimization
        batchsize: number of samples sampled in an epoch
        num_mini_batch : batchsize
        lr_actor: learning rate of actor neural network
        lr_critic : learning rate of critic neural network
        max_grad_norm : clip parameters of max gradient norm
        env_maxsteps: max episode length of environment
        curriculum: whether use curriculum learning, only for bipedal controller environment

        '''
        # create folder to save model
        if(not os.path.isdir(save_path)):
            os.mkdir(save_path)

        #writer = SummaryWriter()

        runner =  Runner(env, s_norm, actor, critic, batchsize, 0.99, 0.95, env_maxsteps) #runner to collect data 
        optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor) #actor neural network optimizer
        optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic) # critic neural network optimizer
        num_samples = ppo_epoch* runner.sample_size #total samples
        num_sample = 0 #current samples
       

        if(curriculum):
        #important settings for curriculum learning
            rwd_threshold = 170 # if reward is larger than this threshold, shrink the kp and kd of assistance torque
            full_assist_flag = True


        for i in range(ppo_epoch):
            num_sample += runner.sample_size           
            rollouts = runner.run() # get collected samples
            obs = rollouts["obs"] 
            acs = rollouts["acs"]
            obs_next = rollouts["obs_next"] 
            rwds = rollouts["rwds"]
            dones = rollouts["dones"] 
            vtars = rollouts["vtars"]
            advs = rollouts["advs"]
            alogps = rollouts["alogps"]
            vpreds = rollouts["vpreds"]

            #normalize advantages
            advs = (advs-advs.mean())/(advs.std() + 0.0001)
            advs = np.clip(advs, -4, 4)

            value_loss_epoch = 0 #loss of critic network
            action_loss_epoch = 0 #loss of actor network
            symmetry_loss_epoch = 0 #loss of symmetry

            num_iter = int(obs.shape[0] / num_mini_batch) 

            for iter_inner in range(10):

                id_list = np.arange(obs.shape[0])
                np.random.shuffle(id_list) #shuffle data

                for idx in range(num_iter):
                    idx_range  = np.random.choice(obs.shape[0], num_mini_batch, replace=False)
                    obs_batch = torch.Tensor(obs[idx_range, :]).float()   
                    acs_batch = torch.Tensor(acs[idx_range, :]).float()
                    vtars_batch = torch.Tensor(vtars[idx_range, :]).float()
                    alogps_old_batch = torch.Tensor(alogps[idx_range, :]).float()
                    advs_batch = torch.Tensor(advs[idx_range, :]).float()
                    vpreds_old_batch = torch.Tensor(vpreds[idx_range, :]).float()

                    optimizer_actor.zero_grad()
                    optimizer_critic.zero_grad()

                    obs_normed_batch = s_norm(obs_batch)
                    m = actor.act_distribution(obs_normed_batch)
                    alogps_batch = m.log_prob(acs_batch).sum(dim=1).view(-1,1)
                    vpreds_batch = critic(obs_normed_batch) 
                    ratio = torch.exp(alogps_batch -alogps_old_batch)
                    

                    #add your code here, compute objective function in PPO
                    #hint: you can use torch.clamp() to clip objective functions
                    rateadvs = ratio * advs_batch
                    cliprate = torch.clamp(ratio, 1.0 - clip_param,1.0 + clip_param) * advs_batch 
                    objective = torch.mean(rateadvs, cliprate)

                    clip_vpreds_batch = vpreds_old_batch + (vpreds_batch-vpreds_old_batch).clamp(-clip_param, clip_param)
                    clipvalue = 0.5 * ((clip_vpreds_batch - vtars_batch).pow(2)/(critic.v_std)).mean()
                    value_loss = 0.5 * ((vpreds_batch - vtars_batch).pow(2)/(critic.v_std)).mean()
                    value_loss = torch.min(value_loss, clipvalue)
                

                

                    objective = -objective
                    value_loss.backward()
                    objective.backward()

                    nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                    nn.utils.clip_grad_norm_(critic.parameters(),max_grad_norm)

                    optimizer_actor.step()
                    optimizer_critic.step()


                    

                
                

            num_updates = num_iter * num_mini_batch
           
            if(i%10==0):
              data = {"actor": actor.state_dict(),
              "critic": critic.state_dict(),
              "s_norm": s_norm.state_dict()}

              torch.save(data, save_path+"/checkpoint_"+str(i)+".tar")
            if(i%5==0):
              print("epoch:{}".format(i))
              rwd_acc, test_step= runner.testModel( )
              if(curriculum):
                if(rwd_acc>rwd_threshold):
                    # shrink kp and kd
                    if(full_assist_flag == True):
                        full_assist_flag = False
                        rwd_threshold*=0.8
                    adjust_KpandKd(runner.env, 0.75)
                print("kp_start:{}".format(runner.env.kpandkd_start))
                print("kp_end:{}".format(runner.env.kpandkd_end))
              print(" ")

def actor_critic(save_path,
        env,
        actor,
        critic,
        s_norm,
        actor_critic_epoch,
        batchsize,
        num_mini_batch,
        lr_actor=None,
        lr_critic=None,
        max_grad_norm=None,
        env_maxsteps = 500,
        curriculum = True
        
        ):
        '''
        simple actor critic optimzation algorithm
        env: simulation environment
        actor: actor neural network
        critic: critic network
        s_norm : state normalization module
        ppo_epoch : epochs of PPO optimization
        batchsize: number of samples sampled in an epoch
        num_mini_batch : batchsize
        lr_actor: learning rate of actor neural network
        lr_critic : learning rate of critic neural network
        max_grad_norm : clip parameters of max gradient norm
        env_maxsteps: max episode length of environment
        curriculum: whether use curriculum learning, only for bipedal controller environment

        '''
        # create folder to save model
        if(not os.path.isdir(save_path)):
            os.mkdir(save_path)
        runner =  Runner(env, s_norm, actor, critic, batchsize, 0.99, 0.95, env_maxsteps) #runner to collect data 
        optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor) #actor neural network optimizer
        optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic) # critic neural network optimizer
        num_samples = actor_critic_epoch* runner.sample_size #total samples
        num_sample = 0 #current samples



        if(curriculum):
        #important settings for curriculum learning
            rwd_threshold = 170 # if reward is larger than this threshold, shrink the kp and kd of assistance torque
            full_assist_flag = True


        for i in range(actor_critic_epoch):
            num_sample += runner.sample_size           
            rollouts = runner.run() # get collected samples
            obs = rollouts["obs"] 
            acs = rollouts["acs"]
            obs_next = rollouts["obs_next"] 
            rwds = rollouts["rwds"]
            dones = rollouts["dones"] 
            vtars = rollouts["vtars"]
            advs = rollouts["advs"]
            alogps = rollouts["alogps"]
            vpreds = rollouts["vpreds"]

            #normalize advantages
            advs = (advs-advs.mean())/(advs.std() + 0.0001)
            advs = np.clip(advs, -4, 4)

            value_loss_epoch = 0 #loss of critic network
            action_loss_epoch = 0 #loss of actor network

            num_iter = int(obs.shape[0] / num_mini_batch) 

            for iter_inner in range(10):

                id_list = np.arange(obs.shape[0])
                np.random.shuffle(id_list) #shuffle data

                for idx in range(num_iter):
                    idx_range  = np.random.choice(obs.shape[0], num_mini_batch, replace=False)
                    obs_batch = torch.Tensor(obs[idx_range, :]).float()   
                    acs_batch = torch.Tensor(acs[idx_range, :]).float()
                    vtars_batch = torch.Tensor(vtars[idx_range, :]).float()
                    alogps_old_batch = torch.Tensor(alogps[idx_range, :]).float()
                    advs_batch = torch.Tensor(advs[idx_range, :]).float()
                    vpreds_old_batch = torch.Tensor(vpreds[idx_range, :]).float()

                    optimizer_actor.zero_grad()
                    optimizer_critic.zero_grad()

                    obs_normed_batch = s_norm(obs_batch)
                    m = actor.act_distribution(obs_normed_batch)
                    alogps_batch = m.log_prob(acs_batch).sum(dim=1).view(-1,1)
                    vpreds_batch = critic(obs_normed_batch)
                    #compute value_loss
                    # add your code here 
                    value_loss = torch.sum((vpreds_batch - vtars_batch).pow(2)
       
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(),max_grad_norm)
                    optimizer_critic.step()
                    
                    if(iter_inner == 0):
                        #compute your objective here to replace the line below
                        objective = torch.sum(advs_batch*alogps_batch)
                        
                        objective = -objective
                        nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                        objective.backward()
                        optimizer_actor.step()


                    

                

            num_updates = num_iter * num_mini_batch
           
           

            if(i%10==0):
              data = {"actor": actor.state_dict(),
              "critic": critic.state_dict(),
              "s_norm": s_norm.state_dict()}

              torch.save(data, save_path+"/checkpoint_"+str(i)+".tar")
            if(i%5==0):
                print("epoch:{}".format(i))
                rwd_acc, test_steps = runner.testModel( )
                if(curriculum):
                    if(rwd_acc>rwd_threshold):
                # shrink kp and kd
                        if(full_assist_flag == True):
                            full_assist_flag = False
                            rwd_threshold*=0.8
                        adjust_KpandKd(runner.env, 0.75)
                    print("kp_start:{}".format(runner.env.kpandkd_start))
                    print("kp_end:{}".format(runner.env.kpandkd_end))
                print(" ")

def reinforce(save_path,
        env,
        actor,
        critic,
        s_norm,
        actor_epoch,
        batchsize,
        num_mini_batch,
        lr_actor=None,
        max_grad_norm=None,
        env_maxsteps = 500,
        curriculum = True
        ):
        '''
        simple actor critic optimzation algorithm
        env: simulation environment
        actor: actor neural network
        critic: critic network
        s_norm : state normalization module
        ppo_epoch : epochs of PPO optimization
        batchsize: number of samples sampled in an epoch
        num_mini_batch : batchsize
        lr_actor: learning rate of actor neural network
        lr_critic : learning rate of critic neural network
        max_grad_norm : clip parameters of max gradient norm
        env_maxsteps: max episode length of environment
        curriculum: whether use curriculum learning, only for bipedal controller environment

        '''
        # create folder to save model
        if(not os.path.isdir(save_path)):
            os.mkdir(save_path)
        runner =  Runner(env, s_norm, actor, critic, batchsize, 0.99, 0.95, env_maxsteps) #runner to collect data 
        optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor) #actor neural network optimizer
        num_samples = actor_epoch* runner.sample_size #total samples
        num_sample = 0 #current samples

        if(curriculum):
            #important settings for curriculum learning
            rwd_threshold = 170 # if reward is larger than this threshold, shrink the kp and kd of assistance torque
            full_assist_flag = True


        for i in range(actor_epoch):
            num_sample += runner.sample_size           
            rollouts = runner.run() # get collected samples
            obs = rollouts["obs"] 
            acs = rollouts["acs"]
            obs_next = rollouts["obs_next"] 
            rwds = rollouts["rwds"]
            dones = rollouts["dones"] 
            advs = rollouts["advs"]
            alogps = rollouts["alogps"]
            acc_rwds = rollouts["acc_rwds"]

            action_loss_epoch = 0 #loss of actor network

            num_iter = int(obs.shape[0] / num_mini_batch) 

            for iter_inner in range(1):

                id_list = np.arange(obs.shape[0])
                np.random.shuffle(id_list) #shuffle data

                for idx in range(num_iter):
                    idx_range  = np.random.choice(obs.shape[0], num_mini_batch, replace=False)
                    obs_batch = torch.Tensor(obs[idx_range, :]).float()   
                    acs_batch = torch.Tensor(acs[idx_range, :]).float()                
                    alogps_old_batch = torch.Tensor(alogps[idx_range, :]).float()
                    advs_batch = torch.Tensor(advs[idx_range, :]).float()
                    acc_rwds_batch = torch.Tensor(acc_rwds[idx_range, :]).float()

                    optimizer_actor.zero_grad()
                    

                    obs_normed_batch = s_norm(obs_batch)
                    m = actor.act_distribution(obs_normed_batch)
                    alogps_batch = m.log_prob(acs_batch).sum(dim=1).view(-1,1)

                    #add your code here

                    objective = torch.mean(acc_rwds_batch*alogps_batch)
                    

                    

                    objective = - objective
                    objective.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                    optimizer_actor.step()
                    
                

            num_updates = num_iter * num_mini_batch
     

            if(i%10==0):
              data = {"actor": actor.state_dict(),
              "critic":critic.state_dict(),
              "s_norm": s_norm.state_dict()}

              torch.save(data, save_path+"/checkpoint_"+str(i)+".tar")
            if(i%5==0):
              print("epoch:{}".format(i))
              rwd_acc, test_steps = runner.testModel( )
              if(curriculum):
                    if(rwd_acc>rwd_threshold):
                    # shrink kp and kd
                        if(full_assist_flag == True):
                            full_assist_flag = False
                            rwd_threshold*=0.8
                    adjust_KpandKd(runner.env, 0.75)
                    print("kp_start:{}".format(runner.env.kpandkd_start))
                    print("kp_end:{}".format(runner.env.kpandkd_end))
              print(" ")