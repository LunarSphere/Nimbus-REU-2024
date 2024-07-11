import math
import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.env_checker import check_env

class UAVEnv(gym.Env):
    
    
    def __init__(self):
        super(UAVEnv, self).__init__()       
        #important values for mobile edge server 
        self.uav_position = np.array([0.0, 0.0], dtype=np.float32) #x, y position of the UAV
        self.height = 100 #height of the UAV
        self.area = 100 #axa area so this is legnth and width of the area
        self.sum_task_size = 100 * 1048576
        self.bandwidth_nums = 1 
        self.Bandwidth = self.bandwidth_nums * 10 * 6 #Bandwidth 1MHz
        self.p_noisy_los = 10 ** (-13)  #Noise power of LOS channel - 100dBm
        self.p_noisy_nlos = 10 ** (-11)  #Noise power of NLOS channel - 80dBm
        self.flight_speed = 50  #UAV flight speed 50m/s
        self.f_uav = 1.2e9  #UAV edge server frequency = 1.2GHz
        self.f_ue = 2.e8 #User equipment frequency = .6GHz
        self.r = 10 ** (-27) # Influence factors of chip structure on CPU processing
        self.s = 1000 #number of CPU cycles requried for unit bit processing is 1000
        self.p_uplink = 0.1  #Uplink transmission power 0.1W
        self.alpha0 = 1e-5 #reference channel gain at 1m -30dBm
        self.T = 320 #period 320s
        self.t_fly = 1 #flight time
        self.t_com = 7
        self.delta_t = self.t_fly + self.t_com #1s for flight time and 7s for computation time
        self.v_ue = 1 #ue moving speed 1m/s
        self.slot_num = int(self.T/self.delta_t) #number of time slots
        self.m_uav = 9.65 #UAV mass 9.65kg
        self.e_battery_uav = 500000.0 #battery capacity 500kJ #ref: Mobile Edge Computing via a UAV-Mounted Cloudlet: Optimization of Bit Allocation and Path Planning
        self.isinitial = True
        #################### ues ####################
        #important for user equipment (ue)
        self.users = 4 #number of users
        self.block_flag_list = np.random.randint(0,2,self.users, dtype=np.int8) #determine which uavs are blocked or not in line of sight of the drone randomly. 
        #good for random simulation but will definetly change when we using real data. 
        self.loc_ue_list = np.ravel(np.random.uniform(0, 100, size=[self.users, 2])).astype(np.float32)  # Position information: x is random between 0-100
        #print(self.loc_ue_list)debugging delete later
        #important to work with stable baselines below
        self.n_actions = 4
        self.task_list = np.random.randint(2097153, 2621440, self.users)  # Random computing task 2~2.5Mbits -> 80
        
        #will need to think about how to alter below 3 to serve all users at once
        #idea for serving multiple users remove the user selection part of the action space
        #and when calculating the latency loop through each user. 
         #will remove since I'm not using than to define my actions.
        # self.action_bound = [-1, 1]  # Corresponds to the tahn activation function
        # self.action_dim = 4  # The first digit represents the ue id of the service; the middle two digits represent the flight angle and distance; the last 1 digit represents the uninstall rate currently serving the UE
        # self.state_dim = 4 + self.users * 4  # uav battery remain, uav loc, remaining sum task size, all ue loc, all ue task size, all ue block_flag
        self.state = {}
        #initliaze uav battery remaining, uav location, sum of task size, all ue location, all ue task size, all ue block flag
        self.start_state = self.state # the changes I implemented have changed the output of the state. to not be float 32 7/10/2024
        #print(self.state) debugging delete later
        #decided to change action space to normalize values between -1 and 1 and have a tuple with discrete number of users. 
        self.action_space = spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)
        
        self.observation_space = spaces.Dict({
            'battery_remaining': spaces.Box(low=0, high=self.e_battery_uav, shape=(1,), dtype=np.float32),
            'uav_location': spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32),  # Assuming normalized coordinates
            'ue_locations': spaces.Box(low=0, high=100, shape=(2*self.users,), dtype=np.float32),
            'ue_tasksize': spaces.Box(low=2097153, high=2621440, shape=(self.users,), dtype=np.int64),  # Assuming values range between 0 and 10
            'ue_blocklist': spaces.MultiBinary(self.users)
        })

    def modify_state(self):
            #experimental state modification
            self.state = {
                'battery_remaining': np.array([self.e_battery_uav], dtype=np.float32),
                'uav_location': self.uav_position,
                'ue_locations': self.loc_ue_list,
                'ue_tasksize': self.task_list,
                'ue_blocklist': self.block_flag_list
                }
            #array looks like
            #[battery remaining, uavlocation x, uav location y, 
            # sum of task size, ue1 x, ue1 y, ue2 x, ue2 y, ...,
            # ue1 task size, ue2 task size, ..., ue1 block flag, ...]
            #if self.isinitial == True:

                #self.isinitial = False
            # else:
            #     state = np.append(self.e_battery_uav, self.uav_position)
            #     state = np.append(state, self.sum_task_size)
            #     state = np.append(state, np.ravel(self.loc_ue_list))
            #     state = np.append(state, self.task_list)
            #     state = np.append(state, self.block_flag_list)
            # return self.state


        
        #reset is finished in theory; possible fault point is the modify_state method and not splitting into multiple methods 
        #as shown in the sample code. 7/10/2024
    def reset(self, seed = None, options = None):
        super().reset(seed=seed, options=options)
        #default settings for battery and other values
        self.sum_task_size = 100 * 1048576
        self.e_battery_uav = 500000.0
        self.uav_position = np.array([0.0, 0.0], dtype=np.float32)
        self.loc_ue_list = np.ravel(np.random.uniform(0, 100, size=[self.users, 2])).astype(np.float32)  # Position information: x is random between 0-100
        #where reset step would be
        self.task_list = np.random.randint(2097153, 2621440, self.users)  # Random computing task 2~2.5Mbits -> 80
        self.block_flag_list = np.random.randint(0,2,self.users,dtype=np.int8) #determine which uavs are blocked or not in line of sight of the drone randomly.
        #update state to reflect changes
        self.modify_state()
        #print(self.state) debugging delete later 

        #below commands are necessary for stable baselines 3 to work
        info = {}
        self.modify_state()
        observation = self.state #self.modify_state(self.state)
        #print(observation) debugging delete later
        return (observation, info)
        #return np.array([self.uav_position]).astype(np.float32), {}





    #7/9/2024 not finished with above methods for reset but I want to switch to step method 
    #will return to top tasks tomorrow. 
    #should spend some time thinking about novelty. 
    def step(self,action):
        step_redo = False
        terminated = False
        offloading_ratio_change = False
        reset_dist = False

        # get values from the action
        angle = action[0]
        flight_distance = action[1]
        climb_angle = action[2]
        user_selection = action[3]
        offloading_ratio = (action[4] + 1) /2
        
        # convert to actual values
        angle = (angle + 1) * 180  # Map [-1, 1] to [0, 360]
        flight_distance = (flight_distance + 1) * 49.5  # Map [-1, 1] to [0, 99]
        climb_angle = climb_angle * 45  # Assuming climb angle range is [-45, 45] degrees
        
        # Convert user selection action to a discrete integer in the range [0, max_users - 1]
        user_selection = int((user_selection + 1) * (self.users - 1) / 2)
        task_size = self.task_list[user_selection]
        block_flag = self.block_flag_list[user_selection]

        #calculate chosen flight distance
        dis_fly = flight_distance * self.flight_speed * self.t_fly
        #calculate energy consumption
        e_fly = (dis_fly/self.t_fly) ** 2 * self.m_uav *  self.t_fly * 0.5
        #calculate post flight position of uav
        dx_uav = dis_fly * math.cos(angle)
        dy_uav = dis_fly * math.sin(angle)
        loc_uav_post_x = self.uav_position[0] + dx_uav
        loc_uav_post_y = self.uav_position[1] + dy_uav
        
        #calculate server computing energy consumption
        t_server = offloading_ratio * task_size / (self.f_uav / self.s) 
        e_server = self.r * self.s ** 3 * t_server

        #logic to determine how step progresses. 
        if self.sum_task_size == 0:
            terminated = True
            reward = 0
        elif self.sum_task_size - self.task_list[user_selection] < 0:
            self.task_list = np.ones(self.users) * self.sum_task_size
            reward = 0
            step_redo = True
        elif loc_uav_post_x < 0 or loc_uav_post_x > self.area or loc_uav_post_y < 0 or loc_uav_post_y > self.area:  # uav position is wrong:
            reset_dist = True
            ue_coords = [self.loc_ue_list[user_selection*2], self.loc_ue_list[(user_selection*2)+1]]
            delay = self.com_delay(ue_coords, self.uav_position, offloading_ratio, task_size, block_flag)
            reward = -delay
            # Update status at next moment
            self.e_battery_uav = self.e_battery_uav - e_server  # uav remaining power
            self.reset2(delay, self.uav_position[0], self.uav_position[1], offloading_ratio, task_size, user_selection)
        elif self.e_battery_uav < e_fly or self.e_battery_uav - e_fly < e_server:  # uav power cannot support calculations
            ue_coords = [self.loc_ue_list[user_selection*2], self.loc_ue_list[(user_selection*2)+1]]
            delay = self.com_delay(ue_coords, self.uav_position, offloading_ratio, task_size, block_flag)
            reward = -delay
            # Update status at next moment
            self.e_battery_uav = self.e_battery_uav - e_server  # uav remaining power
            self.reset2(delay, self.uav_position[0], self.uav_position[1], 0, task_size, user_selection)
            offloading_ratio_change = True
        else:
            ue_coords = [self.loc_ue_list[user_selection*2], self.loc_ue_list[(user_selection*2)+1]]
            delay = self.com_delay(ue_coords, self.uav_position, offloading_ratio, task_size, block_flag)
            reward = -delay
            # Update status at next moment
            self.e_battery_uav = self.e_battery_uav - e_fly - e_server  # uav remaining power
            self.uav_position[0] = loc_uav_post_x
            self.uav_position[1] = loc_uav_post_y
            self.reset2(delay, loc_uav_post_x, loc_uav_post_y, offloading_ratio, task_size, user_selection)
        #below is necessary for the environment to work with stable baselines 3 but I will need to change it later
        reward = 1 if self.uav_position[0] == 0 else 0
        
        truncated = False #means step ended due to a time limit
        observation = self.state
        info = {"redo_step": step_redo, "reset_dist": reset_dist, "offloading_ratio_change": offloading_ratio_change}
        return (observation, reward, terminated, truncated, info)
    



    def close(self):
        pass

    #beyond this point the methods are not necessary for the environment to work 
    #but are necessary for our simulation
    #above this point the methods are necessary to pass the stable baselines check
     
     
      #Reset ue task size, remaining total task size, ue position, and record to file
    def reset2(self, delay, x, y, offloading_ratio, task_size, user_selection):
        #ue_coords = [self.loc_ue_list[user_selection*2], self.loc_ue_list[(user_selection*2)+1]]
        self.sum_task_size -= self.task_list[user_selection]  # Remaining tasks
        for i in range(self.users):  #ue position after random movement
            tmp = np.random.rand(2)
            theta_ue = tmp[0] * np.pi * 2  #ue Random movement angle
            dis_ue = tmp[1] * self.delta_t * self.v_ue  #ue random movement distance
            self.loc_ue_list[user_selection*2] = self.loc_ue_list[user_selection*2] + math.cos(theta_ue) * dis_ue
            self.loc_ue_list[(user_selection*2)+1] = self.loc_ue_list[(user_selection*2)+1] + math.sin(theta_ue) * dis_ue
            self.loc_ue_list[i] = np.clip(self.loc_ue_list[i], 0, self.area)
        #self.reset_step()  # ue random computing task 1~2Mbits # 4 ue, ue occlusion situation
        #where reset step would be
        self.task_list = np.random.randint(2097153, 2621440, self.users)  # Random computing task 2~2.5Mbits -> 80
        self.block_flag_list = np.random.randint(0,2,self.users,dtype=np.int8) #determine which uavs are blocked or not in line of sight of the drone randomly.
        # Record UE expenses
        file_name = 'output.txt'
        # file_name = 'output_ddpg_' + str(self.bandwidth_nums) + 'MHz.txt'
        with open(file_name, 'a') as file_obj:
            file_obj.write("\nUE-" + '{:d}'.format(user_selection) + ", task size: " + '{:d}'.format(int(task_size)) + ", offloading ratio:" + '{:.2f}'.format(offloading_ratio))
            file_obj.write("\ndelay:" + '{:.2f}'.format(delay))
            file_obj.write("\nUAV hover loc:" + "[" + '{:.2f}'.format(x) + ', ' + '{:.2f}'.format(y) + ']')  # Output retains two digits of the result

           
    # Calculate cost
    #loc_uav == to uav_position and loc_ue == to ue_position
    def com_delay(self, loc_ue, loc_uav, offloading_ratio, task_size, block_flag):
        #print(loc_uav)
        #print (loc_ue) #debugging uav position is just a single value
        dx = loc_uav[0] - loc_ue[0]
        dy = loc_uav[1] - loc_ue[1]
        dh = self.height
        dist_uav_ue = np.sqrt(dx * dx + dy * dy + dh * dh)
        p_noise = self.p_noisy_los
        if block_flag == 1:
            p_noise = self.p_noisy_nlos
        g_uav_ue = abs(self.alpha0 / dist_uav_ue ** 2)  #Channel gain
        trans_rate = self.Bandwidth * math.log2(1 + self.p_uplink * g_uav_ue / p_noise) # Uplink transmission rate bps
        t_tr = offloading_ratio * task_size / trans_rate  # Upload delay, 1B=8bit
        t_edge_com = offloading_ratio * task_size / (self.f_uav / self.s)  # Calculate the delay on the UAV edge server
        t_local_com = (1 - offloading_ratio) * task_size / (self.f_ue / self.s)  # Local calculation delay
        if t_tr < 0 or t_edge_com < 0 or t_local_com < 0:
            raise Exception(print("+++++++++++++++++!! error !!+++++++++++++++++++++++"))
        return max([t_tr + t_edge_com, t_local_com])  #Flight time impact factor

if __name__ == "__main__":
    env = UAVEnv()
    #obs, _ = env.reset()
    #print(obs)
    check_env(env)
    