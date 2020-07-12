import multiprocessing as mp
import numpy as np
import time
import random

class GenRandom(mp.Process):
    def __init__(self, id):
        mp.Process.__init__(self)
        self.id = id
        
    def run(self):
        np.random.seed(self.id)
        print(self.id, np.random.rand(2))


swarm = [GenRandom(i) for i in range(4)]
for proc in swarm:
    proc.start()

