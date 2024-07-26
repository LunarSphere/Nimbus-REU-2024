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

        #################### ues ####################
        #important for user equipment (ue)
        self.users = 999 #number of users hard coded for now
        self.block_flag_list = np.random.randint(0,2,self.users, dtype=np.int8) #determine which uavs are blocked or not in line of sight of the drone randomly. 
        self.task_list = np.random.randint(2097153, 2621440, self.users)  # Random computing task 2~2.5Mbits -> 80
        self.active_list = np.zeros(self.users) #determine if user is actively transmitting and receiving data
        #good for random simulation but will definetly change when we using real data. 
        self.loc_ue_list = np.ravel(np.random.uniform(0, 100, size=[self.users, 3])).astype(np.float32)  # Position information: x is random between 0-100
        for i in range(self.users):
            self.loc_ue_list[(i*2)+2] = 0; #z position of ue is 0
        
        #################### data ####################
                # Read the csv skippping every 30th row mimicking 30 frames per second  
        self.df = pd.read_csv('00_tracks.csv', usecols=['frame', 'trackId', 'xCenter', 'yCenter'])
        # sort by frame 
        self.df = self.df.sort_values(by=['frame', 'trackId']).reset_index(drop=True)
        self.frame = 0 #frame counter for df
        self.steps = 0 #step counter for environment
        
        #################### stable baselines ####################
        #important to work with stable baselines below
        self.state = {}
        self.state = self.modify_state()
        #initliaze uav battery remaining, uav location, sum of task size, all ue location, all ue task size, all ue block flag
        self.start_state = self.state
        #decided to change action space to normalize values between -1 and 1 and have a tuple with discrete number of users. 
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        
        self.observation_space = spaces.Dict({
            'uav_location': spaces.Box(low=0, high=100, shape=(3,), dtype=np.float32),  # Assuming normalized coordinates
            'ue_locations': spaces.Box(low=0, high=100, shape=(3*self.users,), dtype=np.float32),
            'ue_active': spaces.MultiBinary(self.users) #determine if user is actively transmitting and receiving data
        })

    def normalize(self,value, old_min, old_max, new_min, new_max):
        return new_min + (value - old_min) * (new_max - new_min) / (old_max - old_min)

    def modify_state(self):
            #experimental state modification
            self.state = {
                'uav_location': self.uav_position,
                'ue_locations': self.loc_ue_list,
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
        self.active_list = np.zeros(self.users, dtype=np.int8) #determine if user is actively transmitting and receiving data
        #print(action)
        #print(self.state)
        terminated = False #episode ends

        movement = self.v_ue * action
        self.uav_position = self.uav_position + movement
        self.uav_position_z = np.clip(self.uav_position[2] + movement[2], 20, 100) #z position of uav is 20 to 100
        self.uav_position_x_y = np.clip(self.uav_position + movement, 0, 100) 
        self.uav_position = np.array([self.uav_position_x_y[0], self.uav_position_x_y[1], self.uav_position_z], dtype=np.float32)      

        # perform chosen actions 
        #30 steps means means each episode is about 1 second in real time
        if self.steps == 30: #terminate at end of period #add a case for power running out in future
            terminated = True
            self.structured_update(self.df)
            delay = self.transmission_delay(self.uav_position, self.loc_ue_list, self.active_list)
            #print(delay)
            reward = 1/min(delay) #trains to minimize reward maximizing the minimum delay
            #reward = np.exp(-max(delay)) #trains to maximize reward minimizing the maximum delay
            #print(reward)
        else: #if not terminated and sufficient conditions
            self.structured_update(self.df)
            delay = self.transmission_delay(self.uav_position, self.loc_ue_list, self.active_list)
            #print(delay)
            reward = 1/min(delay) #working for minimizing the max delay per episode
            #reward = np.exp(-max(delay)) #trains to maximize reward minimizing the maximum delays
            #print(reward)
            
            #self.random_update(max(delay), self.uav_position[0], self.uav_position[1], self.uav_position[2], power)
        self.frame += 30
        self.steps += 1
        # Reset the frame counter if it exceeds the maximum
        if self.frame >= (self.df['frame'].max() - 90):
            done = True
            self.frame  = 0
            #generate random number 00 - 31
            random_number = random.randint(0, 32)
            self.df = pd.read_csv(f'{random_number:02}_tracks.csv', usecols=['frame', 'trackId', 'xCenter', 'yCenter'])
            # sort by frame 
            self.df = self.df.sort_values(by=['frame', 'trackId']).reset_index(drop=True)
            self.structured_update(self.df)
            print(f"Switched to {random_number:02}_tracks.csv")
        truncated = False #means step ended due to a time limit
        self.modify_state()
        observation = self.state
        #print uav position, ue position, offloading ratio, task size, reward, terminated, truncated
        #print("UAV position: ", self.uav_position)
        #print(reward)
        info = {'delay': delay}
        return (observation, reward, terminated, truncated, info)
    
    def close(self):
        pass

    #beyond this point the methods are not necessary for the environment to work 
    #but are necessary for our simulation
    #above this point the methods are necessary to pass the stable baselines check
    def structured_update(self, df):
        """
        This function updates the positions of users in the environment based on the user locations
        from CSV files in the csv_files folder.
        it also updates which users are active and servicable.
        this version of structured update is updated to not use a for loop and instead use vectorized operations
        """
        df = df[df['frame'] % 30 == 0].copy()  # Skip every frame not divisible by 30 and make a copy

        # Normalize the x and y coordinates using vectorized operations
        df['xCenter'] = (df['xCenter'] - df['xCenter'].min()) * (100 / (df['xCenter'].max() - df['xCenter'].min()))
        df['yCenter'] = (df['yCenter'] - df['yCenter'].min()) * (100 / (df['yCenter'].max() - df['yCenter'].min()))

        # Select only the rows for the current frame
        current_frame_df = df[df['frame'] == self.frame]

        # Update the user locations and active status using vectorized operations
        track_ids = current_frame_df['trackId'].values
        self.loc_ue_list[track_ids * 2] = current_frame_df['xCenter'].values
        self.loc_ue_list[track_ids * 2 + 1] = current_frame_df['yCenter'].values
        self.active_list[track_ids] = 1




    #loc_uav == to uav_position and loc_ue == to ue_position
    def transmission_delay(self, uav_position, ue_pos_list, active_flag_list):
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
        # return list of propagation delays and use the max delay as the reward
        ue_propagation_delay_list = []
        #print(active_flag_list)
        for ue in range(self.users):    
            if active_flag_list[ue] == 1: #if user is active
                ue_position = np.array([ue_pos_list[ue*2], ue_pos_list[(ue*2)+1], 0])


                distance = np.linalg.norm(uav_position - ue_position) #meters
                if distance == 0:
                    distance = 1e-6  # Prevent division by zero
                propagation_delay = distance / self.speed_light #seconds
                #print(distance)
                ue_propagation_delay_list.append(propagation_delay)
        #print(self.frame)
        #print(active_flag_list)
        #print(ue_propagation_delay_list)      
        return ue_propagation_delay_list

    #Reset ue task size, remaining total task size, ue position, and record to file. 
    # this function serves as a means of updating the simulation environment
    def random_update(self, delay_list, x, y, z):
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
        self.active_list = np.random.randint(0,2,self.users,dtype=np.int8) #determine if user is actively transmitting and receiving data
        # Record UE expenses
        file_name = 'output.csv'
        with open(file_name, 'a', newline='') as file_obj:
            csv_writer = csv.writer(file_obj)

            # Write the header only if the file is empty
            file_obj.seek(0, 2)  # Move the cursor to the end of the file
            if file_obj.tell() == 0:
                csv_writer.writerow(["Delay_List", "UAV Hover Location X", "UAV Hover Location Y", "UAV Hover Location Z"])
            
            # Write the data
            csv_writer.writerow([
                '{:.8f}'.format(delay_list),
                '{:.2f}'.format(x),
                '{:.2f}'.format(y),
                '{:.2f}'.format(z),
            ])

                # def structured_update(self, df): 
    #     """
    #     This function is used to update the update the positions of users in an environment
    #     based on the user locations of csv files in the csv_files folder.  
    #     """
    #     df = df.copy() # make a copy of the data frame
    #     # skip every frame not divisible by 30
    #     df = df[df['frame'] % 30 == 0]
    #     #df.to_csv('output1.csv') #debugging purposes
    #     #normalize the x and y coordinates
    #     df['xCenter'] = df['xCenter'].apply(lambda x: self.normalize(x, df['xCenter'].min(), df['xCenter'].max(), 0, 100))
    #     df['yCenter'] = df['yCenter'].apply(lambda y: self.normalize(y, df['yCenter'].min(), df['yCenter'].max(), 0, 100))
    #     # Max number of users
    #     Max_users = df['trackId'].max()
    #     #filter_df = df[df['frame'] == self.global_csv_tracker]
    #     #subset_df = df.iloc[self.global_csv_tracker:]
    #     for index, row in df.iterrows():
    #         if row['frame'] == self.frame:
    #             #print(row['trackId'], row['xCenter'], row['yCenter'])
    #             track_id = int(row['trackId'])
    #             self.loc_ue_list[(track_id)*2] = row['xCenter']
    #             self.loc_ue_list[(track_id-1)*2+1] = row['yCenter']
    #             self.active_list[(track_id)-1] = 1
                
    #         if self.frame >= 20000:
    #             self.frame = 0
    #     # print(Max_users)
    #     # print(df['xCenter'].min(), df['xCenter'].max())
    #     # print(df['yCenter'].min(), df['yCenter'].max())




if __name__ == "__main__":
    env = UAVEnv()
    #env.structured_update()
    check_env(env)
    
