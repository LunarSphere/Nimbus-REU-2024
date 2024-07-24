import math
import random
import pandas as pd
import csv
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.env_checker import check_env

class UAVEnv(gym.Env):
    def __init__(self):
        super(UAVEnv, self).__init__()       
        #################### uav ####################
        #positioning 
        self.area = 100 #axa area so this is legnth and width of the area
        self.uav_position = np.array([50, 50, 50], dtype=np.float32) #x, y position of the UAV
        self.global_csv_tracker = 0
        #data transmission 
        #self.p_noisy_los = 10 ** (-13)  #Noise power of LOS channel - 100dBm
        #self.p_noisy_nlos = 10 ** (-11)  #Noise power of NLOS channel - 80dBm
        #goal is to minimize transmission delay
        #change cpu frequency to transmission frequency
        #compute transmission delay between uav and ue
        # uav and user should both be 28ghz
        # if its determined by distance over speed of light then what is role of path loss and user frequency
        self.steps = 0
        self.flight_speed = 10  #UAV flight speed 50m/s
        self.t_fly = 1 #flight time
        self.delta_t = self.t_fly #1s for flight time and 7s for computation time
        self.v_ue = 10 #ue moving speed 10m/s
        self.m_uav = 9.65 #UAV mass 9.65kg
        #self.total_power = 500000.0 #battery capacity 500kJ #ref: Mobile Edge Computing via a UAV-Mounted Cloudlet: Optimization of Bit Allocation and Path Planning
        #################### calculate transmission ####################
        self.f_uav_basestation = self.f_ue_comm = 28e9  #UAV and base station frequency 28GHz
        self.Bandwidth = 100e6  #Bandwidth 100MHz
        self.speed_light = 3 * 10 ** 8  #Speed of light 3*10^8m/s
        self.total_power = 1000 #assume total power for tranmission is 400W

        #################### ues ####################
        #important for user equipment (ue)
        self.users = 383 #number of users hard coded for now
        self.block_flag_list = np.random.randint(0,2,self.users, dtype=np.int8) #determine which uavs are blocked or not in line of sight of the drone randomly. 
        self.task_list = np.random.randint(2097153, 2621440, self.users)  # Random computing task 2~2.5Mbits -> 80
        self.active_list = np.zeros(self.users) #determine if user is actively transmitting and receiving data
        #good for random simulation but will definetly change when we using real data. 
        self.loc_ue_list = np.ravel(np.random.uniform(0, 100, size=[self.users, 3])).astype(np.float32)  # Position information: x is random between 0-100
        for i in range(self.users):
            self.loc_ue_list[(i*2)+2] = 0; #z position of ue is 0
        
        #################### stable baselines ####################
        #important to work with stable baselines below
        self.state = {}
        self.state = self.modify_state()
        #initliaze uav battery remaining, uav location, sum of task size, all ue location, all ue task size, all ue block flag
        self.start_state = self.state
        #decided to change action space to normalize values between -1 and 1 and have a tuple with discrete number of users. 
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        
        self.observation_space = spaces.Dict({
            'power_remaining': spaces.Box(low=0, high=self.total_power, shape=(1,), dtype=np.float32),
            'uav_location': spaces.Box(low=0, high=100, shape=(3,), dtype=np.float32),  # Assuming normalized coordinates
            'ue_locations': spaces.Box(low=0, high=100, shape=(3*self.users,), dtype=np.float32),
            'ue_data_request_size': spaces.Box(low=1e9, high=5e9, shape=(self.users,), dtype=np.int64),  # Assuming values range between 0 and 10
            'ue_blocklist': spaces.MultiBinary(self.users), #determine which uavs are blocked or not in line of sight of the drone randomly.
            'ue_active': spaces.MultiBinary(self.users) #determine if user is actively transmitting and receiving data
        })

    def normalize(self,value, old_min, old_max, new_min, new_max):
        return new_min + (value - old_min) * (new_max - new_min) / (old_max - old_min)

    def modify_state(self):
            #experimental state modification
            self.state = {
                'power_remaining': np.array([self.total_power], dtype=np.float32),
                'uav_location': self.uav_position,
                'ue_locations': self.loc_ue_list,
                'ue_data_request_size': self.task_list,
                'ue_blocklist': self.block_flag_list,
                'ue_active': self.active_list
                }
            
    #reset environment for next episode
    def reset(self, seed = None, options = None):
        super().reset(seed=seed, options=options)
        #default settings for battery and other values
        #self.step_count = 0
        self.steps = 0
        self.total_power = 1000 #4000 #(watts)
        self.uav_position = np.array([50, 50, 50], dtype=np.float32)
        self.loc_ue_list = np.ravel(np.random.uniform(0, 100, size=[self.users, 3])).astype(np.float32)  # Position information: x is random between 0-100
        #where reset step would be
        self.task_list = np.random.randint(1e9, 5e9, self.users, dtype=np.int64)  # Random computing task 2~2.5Mbits -> 80
        self.block_flag_list = np.random.randint(0,2,self.users,dtype=np.int8) #determine which uavs are blocked or not in line of sight of the drone randomly.
        self.active_list = np.random.randint(0,2,self.users,dtype=np.int8) #determine if user is actively transmitting and receiving data
        #update state to reflect changes
        #self.modify_state()

        #below commands are necessary for stable baselines 3 to work
        info = {}
        self.modify_state()
        observation = self.state 
        return observation, info





    #7/9/2024 not finished with above methods for reset but I want to switch to step method 
    #will return to top tasks tomorrow. 
    #should spend some time thinking about novelty. 
    def step(self,action):
        self.active_list = np.random.randint(0,2,self.users,dtype=np.int8)
        #print(action)
        #print(self.state)
        terminated = False #episode ends
        self.steps += 1
        movement = self.v_ue * action[0:3]
        self.uav_position = np.clip(self.uav_position + movement, 0, 100)
        


        # # get values from the action
        # angle = action[0]
        # flight_distance = action[1]
        # climb_angle = action[2]
        power = action[3]

        
        # convert to actual values
        # angle = (angle + 1) * 180  # Map [-1, 1] to [0, 360]
        # flight_distance = (flight_distance + 1) * 49.5  # Map [-1, 1] to [0, 99]
        # climb_angle = climb_angle * 45  # Assuming climb angle range is [-45, 45] degrees
        
        # Convert user selection action to a discrete integer in the range [0, max_users - 1]
        power = (power + 1) * (self.total_power - 1) / 2

        #calculate chosen flight distance
        #dis_fly = flight_distance * self.flight_speed * self.t_fly

        #calculate post flight position of uav
        # dx_uav = dis_fly * math.cos(angle)
        # dy_uav = dis_fly * math.sin(angle)
        # loc_uav_post_x = np.clip(self.uav_position[0] + dx_uav, 0, self.area)
        # loc_uav_post_y = np.clip(self.uav_position[1] + dy_uav, 0, self.area)
        # perform chosen actions 
        if self.steps == 20: #terminate at end of period #add a case for power running out in future
            terminated = True
            delay = self.transmission_delay(self.uav_position, self.loc_ue_list, self.block_flag_list, self.active_list, self.task_list, power)
            reward = -1/max(delay)
            #reward = -max(delay)
            #print(reward)
        else: #if not terminated and sufficient conditions
            delay = self.transmission_delay(self.uav_position, self.loc_ue_list, self.block_flag_list, self.active_list, self.task_list, power)
            reward = -1/max(delay)
            #reward = -max(delay)
            #print(reward)
            self.structured_update()
            #self.random_update(max(delay), self.uav_position[0], self.uav_position[1], self.uav_position[2], power)
        truncated = False #means step ended due to a time limit
        self.modify_state()
        observation = self.state
        #print uav position, ue position, offloading ratio, task size, reward, terminated, truncated
        #print("UAV position: ", self.uav_position)
        info = {}
        return (observation, reward, terminated, truncated, info)
    
    def close(self):
        pass

    #beyond this point the methods are not necessary for the environment to work 
    #but are necessary for our simulation
    #above this point the methods are necessary to pass the stable baselines check

    def structured_update(self): 
        """
        This function is used to update the update the positions of users in an environment
        based on the user locations of csv files in the csv_files folder.  
        """
        # Read the csv skippping every 30th row mimicking 30 frames per second  
        df = pd.read_csv('csv_files/00_tracks.csv', usecols=['frame', 'trackId', 'xCenter', 'yCenter'])
        # sort by frame 
        df = df.sort_values(by=['frame', 'trackId']).reset_index(drop=True)

        # skip every frame not divisible by 30
        df = df[df['frame'] % 30 == 0]
        df.to_csv('output1.csv') #debugging purposes
        #normalize the x and y coordinates
        df['xCenter'] = df['xCenter'].apply(lambda x: self.normalize(x, df['xCenter'].min(), df['xCenter'].max(), 0, 100))
        df['yCenter'] = df['yCenter'].apply(lambda y: self.normalize(y, df['yCenter'].min(), df['yCenter'].max(), 0, 100))
        # Max number of users
        Max_users = df['trackId'].max()
        filter_df = df[df['frame'] == self.global_csv_tracker]
        #subset_df = df.iloc[self.global_csv_tracker:]
        for index, row in filter_df.iterrows():
            #print(row['trackId'], row['xCenter'], row['yCenter'])
            track_id = int(row['trackId'])
            self.loc_ue_list[(track_id)*2] = row['xCenter']
            self.loc_ue_list[(track_id-1)*2+1] = row['yCenter']
            self.active_list[(track_id)-1] = 1
            
            if self.global_csv_tracker >= 20000:
                self.global_csv_tracker = 0
        self.global_csv_tracker += 30
        # print(Max_users)
        # print(df['xCenter'].min(), df['xCenter'].max())
        # print(df['yCenter'].min(), df['yCenter'].max())

    #Reset ue task size, remaining total task size, ue position, and record to file. 
    # this function serves as a means of updating the simulation environment
    def random_update(self, delay_list, x, y, z, power):
        for i in range(self.users):  #ue position after random movement
            tmp = np.random.rand(2)
            theta_ue = tmp[0] * np.pi * 2  #ue Random movement angle
            dis_ue = tmp[1] * self.delta_t * self.v_ue  #ue random movement distance
            # Update ue position fixed the out of bounds issue 
            self.loc_ue_list[i*2] = self.loc_ue_list[i*2] + math.cos(theta_ue) * dis_ue
            self.loc_ue_list[(i*2)+1] = self.loc_ue_list[(i*2)+1] + math.sin(theta_ue) * dis_ue
            self.loc_ue_list[i*2] = np.clip(self.loc_ue_list[i*2], 0, self.area)
            self.loc_ue_list[(i*2)+1] = np.clip(self.loc_ue_list[(i*2)+1], 0, self.area)
        #where reset step would be
        self.block_flag_list = np.random.randint(0,2,self.users,dtype=np.int8) #determine which uavs are blocked or not in line of sight of the drone randomly.
        self.active_list = np.random.randint(0,2,self.users,dtype=np.int8) #determine if user is actively transmitting and receiving data
        # Record UE expenses
        file_name = 'output.csv'
        with open(file_name, 'a', newline='') as file_obj:
            csv_writer = csv.writer(file_obj)

            # Write the header only if the file is empty
            file_obj.seek(0, 2)  # Move the cursor to the end of the file
            if file_obj.tell() == 0:
                csv_writer.writerow(["Delay_List", "UAV Hover Location X", "UAV Hover Location Y", "UAV Hover Location Z", "Power"])
            
            # Write the data
            csv_writer.writerow([
                '{:.8f}'.format(delay_list),
                '{:.2f}'.format(x),
                '{:.2f}'.format(y),
                '{:.2f}'.format(z),
                '{:.2f}'.format(power)
                
            ])
            
    
    #loc_uav == to uav_position and loc_ue == to ue_position
    def transmission_delay(self, uav_position, ue_pos_list, block_flag, active_flag, ue_task_list, power_allocated):
        """
        The purpose of this functiion is to calculate the 
        transmission and propagation delay between the UAV and the UE.
        For each user return list of each ues transmission delay
        """
        #delay would be distance divided by speed of light 3*10^8 m/s
        #calculate propagation delay
        #don't need to consider processing delay
        #shorter users are to uav quicker the transmission
        # optimize trajectory
        #

        #################### Possible solution ######################
        # calculate the distance between uav and each ue = dist_uav_ue
        # calculate the propagation delay = dist_uav_ue/speed_of_light
        # Assume the Signal to noise ratio is High 30dBm
        # Assume Bandwidth is 100Mhz since frequency is 28Ghz
        # vary the SNR by Los and Nlos either by calculation or by typicall values
        # use shannon hartley theorem to calculate the transmission rate = Bandwidth * log2(1 + p_uplink * g_uav_ue / p_noise) = Bandwidth * log2(1+SNR)
        # calculate the transmission delay = task_size / transmission rate
        ue_trans_delay_list = []
        for ue in range(self.users):
            if active_flag[ue] == 0:
                
                ue_position = np.array([ue_pos_list[ue*2], ue_pos_list[(ue*2)+1], ue_pos_list[(ue*2)+2]])

                distance = np.linalg.norm(uav_position - ue_position) #meters
                if distance == 0:
                    distance = 1e-6  # Prevent division by zero
                propagation_delay = distance / self.speed_light #1/seconds
                wave_length = self.speed_light / self.f_uav_basestation #wavelength = speed of light / frequency (m/s/GHz)
                #path_loss = (4*np.pi*distance/wave_length) ** 2 #Free space path loss (s/Ghz)^2
                #caclulate SNR  #convert loss to dB
                #recieved_power = power_allocated/ path_loss #divide by 0 error encountered 7/21/2024 (watts)
                #received power = transmit Power + antenna gain - path loss  #recieved power calculation original is incorret
                #noise_power =  10 ** (-174 + 10 * np.log10(self.Bandwidth))

                #################### long noise power calculation ######################
                k = 1.38e-23
                T = 290 #(kelvin)
                noise_power_watts = k * T * self.Bandwidth #thermal noiser power in watts
                noise_figure = 5 #dB
                NF_linear = 10 ** (noise_figure/10) #convert from dB to linear scale
                noise_power = NF_linear * noise_power_watts #total noise power in watts
    
            #noise_power = self.p_noisy_los if block_flag == 0 else self.p_noisy_nlos #may want to consider calculating noise instead of just setting it.
                # snr = 5 dB or 30 dB
                #make snr constant
                snr = 5 #snr = 5db (dB)

                #snr = recieved_power / noise_power
                #if snr == 0:
                #    snr = 1e-6
                #calculate transmission rate using shannon hartley theorem
                transmission_rate = self.Bandwidth * np.log2(1 + snr) #if this or gives weird values look at noise power and converting to linear scale
                #calculate transmission delay
                transmission_delay =  ue_task_list[ue]/ transmission_rate
                #ue_trans_delay_list.append(propagation_delay+transmission_delay)
                ue_trans_delay_list.append(propagation_delay)
        return ue_trans_delay_list
        




if __name__ == "__main__":
    env = UAVEnv()
    #env.structured_update()
    check_env(env)
    
