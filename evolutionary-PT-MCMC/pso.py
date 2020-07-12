
import numpy as np
import random
from random import seed
import time

import os

import copy    # array-copying convenience
import sys     # max float

np.set_printoptions(suppress=True)

################################################################

class Particle(object):
    def __init__(self, num_params, max_limits, min_limits, fitness_function):

        self.fitness_function = fitness_function

        r_pos = np.asarray(random.sample(range(1, num_params+1), num_params) )/ (num_params+1) #to force using random without np and convert to np (to avoid multiprocessing random seed issue)

        np_pos = np.random.rand(num_params)/2 + r_pos/2
        np_vel = np.random.rand(num_params)/2 + r_pos/2

        self.position = ((max_limits - min_limits) * np_pos  + min_limits) # using random.rand() rather than np.random.rand() to avoid multiprocesssing random issues
        self.velocity = ((max_limits - min_limits) * np_vel  + min_limits)
        print('pos', self.position, 'vel', self.velocity)

        self.error =  self.fitness_function(self.position)# curr error
        self.best_part_pos =  self.position.copy()
        self.best_part_err = self.error # best error 


class PSO(object):
    def __init__(self, pop_size, num_params, max_limits, min_limits):
        self.num_params = num_params
        self.pop_size = pop_size
        self.min_limits = min_limits
        self.max_limits = max_limits
        self.swarm, self.best_swarm_pos, self.best_swarm_err = self.create_swarm()
    
    # def fitness_function(self, x):
    #     return None

    def create_swarm(self):
        swarm = [Particle(num_params=self.num_params, max_limits=self.max_limits, min_limits=self.min_limits, fitness_function=self.fitness_function) for i in range(self.pop_size)] 
        best_swarm_pos = [0.0 for i in range(self.num_params)] # not necess.
        best_swarm_err = sys.float_info.max # swarm best
        for i in range(self.pop_size): # check each particle
            if swarm[i].error < best_swarm_err:
                best_swarm_err = swarm[i].error
                best_swarm_pos = copy.copy(swarm[i].position) 
        return swarm, best_swarm_pos, best_swarm_err

    def evolve(self, swarm, best_swarm_pos, best_swarm_err): # this is executed without even calling - due to multi-processing

        w = 0.729    # inertia
        c1 = 1.49445 # cognitive (particle)
        c2 = 1.49445 # social (swarm)
        
        np.random.seed(int(time.time()))
        
        for i in range(self.pop_size): # process each particle 

            r_pos = np.asarray(random.sample(range(1, self.num_params+1), self.num_params) )/ (self.num_params+1) #to force using random without np and convert to np (to avoid multiprocessing random seed issue)

            r1 = np.random.rand(self.num_params)/2 + r_pos/2
            r2 = np.random.rand(self.num_params)

            swarm[i].velocity = ( (w * swarm[i].velocity) + (c1 * r1 * (swarm[i].best_part_pos - swarm[i].position)) +  (c2 * r2 * (best_swarm_pos - swarm[i].position)) )  
            # print('vel', swarm[i].velocity)
            # print('pos', swarm[i].position)

            # for k in range(self.num_params): 
            #     if swarm[i].velocity[k] < self.min_limits[k]:
            #         swarm[i].velocity[k] = self.min_limits[k]
            #     elif swarm[i].velocity[k] > self.max_limits[k]:
            #         swarm[i].velocity[k] = self.max_limits[k]

            swarm[i].position += swarm[i].velocity


            swarm[i].error = self.fitness_function(swarm[i].position)
                
            if swarm[i].error < swarm[i].best_part_err:
                # print('hello')
                swarm[i].best_part_err = swarm[i].error
                swarm[i].best_part_pos = copy.copy(swarm[i].position)

            if swarm[i].error < best_swarm_err:
                # print('hello again')
                best_swarm_err = swarm[i].error
                best_swarm_pos = copy.copy(swarm[i].position)
            else:
                pass
                # print(swarm[i].position, best_swarm_pos)
        
        return swarm, best_swarm_pos, best_swarm_err