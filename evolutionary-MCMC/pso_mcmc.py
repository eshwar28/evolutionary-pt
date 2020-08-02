import numpy as np
import random
import time
import math
import pandas as pd
import os
from pso import PSO, Particle


class Network(object):

    def __init__(self, topology, train_data, test_data, learn_rate = 0.5, alpha = 0.1):
        self.topology = topology  # NN topology [input, hidden, output]
        np.random.seed(int(time.time()))
        self.train_data = train_data
        self.test_data = test_data
        self.W1 = np.random.randn(self.topology[0], self.topology[1]) / np.sqrt(self.topology[0])
        self.B1 = np.random.randn(1, self.topology[1]) / np.sqrt(self.topology[1])  # bias first layer
        self.W2 = np.random.randn(self.topology[1], self.topology[2]) / np.sqrt(self.topology[1])
        self.B2 = np.random.randn(1, self.topology[2]) / np.sqrt(self.topology[1])  # bias second layer
        self.hidout = np.zeros((1, self.topology[1]))  # output of first hidden layer
        self.out = np.zeros((1, self.topology[2]))  # output last layer

    @staticmethod
    def sigmoid(x):
        x = x.astype(np.float128)
        return 1 / (1 + np.exp(-x))

    def sample_er(self, actualout):
        error = np.subtract(self.out, actualout)
        sqerror = np.sum(np.square(error)) / self.topology[2]
        return sqerror

    def calculate_rmse(self, actual, targets):
        return np.sqrt((np.square(np.subtract(np.absolute(actual), np.absolute(targets)))).mean())

    def sample_ad(self, actualout):
        error = np.subtract(self.out, actualout)
        mod_error = np.sum(np.abs(error)) / self.topology[2]
        return mod_error

    def forward_pass(self, X):
        z1 = X.dot(self.W1) - self.B1
        self.hidout = self.sigmoid(z1)  # output of first hidden layer
        z2 = self.hidout.dot(self.W2) - self.B2
        self.out = self.sigmoid(z2)  # output second hidden layer

    def backward_pass(self, Input, desired):
        out_delta = (desired - self.out) * (self.out * (1 - self.out))
        hid_delta = out_delta.dot(self.W2.T) * (self.hidout * (1 - self.hidout))
        self.W2 += (self.hidout.T.dot(out_delta) * self.lrate)
        self.B2 += (-1 * self.lrate * out_delta)
        self.W1 += (Input.T.dot(hid_delta) * self.lrate)
        self.B1 += (-1 * self.lrate * hid_delta)

    def decode(self, w):
        w_layer1_size = self.topology[0] * self.topology[1]
        w_layer2_size = self.topology[1] * self.topology[2]
        w_layer1 = w[0:w_layer1_size]
        self.W1 = np.reshape(w_layer1, (self.topology[0], self.topology[1]))
        w_layer2 = w[w_layer1_size: w_layer1_size + w_layer2_size]
        self.W2 = np.reshape(w_layer2, (self.topology[1], self.topology[2]))
        self.B1 = w[w_layer1_size + w_layer2_size :w_layer1_size + w_layer2_size + self.topology[1]]
        self.B2 = w[w_layer1_size + w_layer2_size + self.topology[1] :w_layer1_size + w_layer2_size + self.topology[1] + self.topology[2]]

    def encode(self):
        w1 = self.W1.ravel()
        w2 = self.W2.ravel()
        w = np.concatenate([w1, w2, self.B1, self.B2])
        return w

    @staticmethod
    def scale_data(data, maxout=1, minout=0, maxin=1, minin=0):
        attribute = data[:]
        attribute = minout + (attribute - minin)*((maxout - minout)/(maxin - minin))
        return attribute

    @staticmethod
    def denormalize(data, indices, maxval, minval):
        for i in range(len(indices)):
            index = indices[i]
            attribute = data[:, index]
            attribute = Network.scale_data(attribute, maxout=maxval[i], minout=minval[i], maxin=1, minin=0)
            data[:, index] = attribute
        return data

    @staticmethod
    def softmax(fx):
        ex = np.exp(fx)
        sum_ex = np.sum(ex, axis = 1)
        sum_ex = np.multiply(np.ones(ex.shape), sum_ex[:, np.newaxis])
        probability = np.divide(ex, sum_ex)
        return probability

    def generate_output(self, data, w):  # BP with SGD (Stocastic BP)
        self.decode(w)  # method to decode w into W1, W2, B1, B2.
        size = data.shape[0]
        Input = np.zeros((1, self.topology[0]))  # temp hold input
        fx = np.zeros((size,self.topology[2]))
        for i in range(0, size):  # to see what fx is produced by your current weight update
            Input = data[i, 0:self.topology[0]]
            self.forward_pass(Input)
            fx[i] = self.out
        return fx

    def evaluate_fitness(self, w):  # BP with SGD (Stocastic BP
        data = self.train_data
        y = data[:, self.topology[0]: self.topology[0] + self.topology[2]]
        fx = self.generate_output(data, w)
        return self.calculate_rmse(fx, y)


class MCMC(PSO, Particle):
    def __init__(self, num_samples, population_size, topology, train_data, test_data, directory, problem_type='regression', max_limit=(10), min_limit=-10):
        self.num_samples = num_samples
        self.pop_size=population_size
        self.topology = topology
        self.train_data = train_data
        self.test_data = test_data
        self.problem_type = problem_type
        self.directory = directory
        self.w_size = (topology[0] * topology[1]) + (topology[1] * topology[2]) + topology[1] + topology[2]
        self.neural_network = Network(topology, train_data, test_data)
        self.min_limits = np.repeat(min_limit, self.w_size)
        self.max_limits = np.repeat(max_limit, self.w_size)
        self.initialize_sampling_parameters()
        self.create_directory(directory)
        PSO.__init__(self, self.pop_size, self.w_size, self.max_limits, self.min_limits,self.neural_network.evaluate_fitness)
        


    def fitness_function(self, x):
        fitness = self.neural_network.evaluate_fitness(x)
        return fitness

    def initialize_sampling_parameters(self):
        self.eta_stepsize = 0.005
        self.wpos_stepsize=0.005
        self.sigma_squared = 25
        self.nu_1 = 0
        self.nu_2 = 0
        self.start = time.time()

    @staticmethod
    def convert_time(secs):
        if secs >= 60:
            mins = str(int(secs/60))
            secs = str(int(secs%60))
        else:
            secs = str(int(secs))
            mins = str(00)
        if len(mins) == 1:
            mins = '0'+mins
        if len(secs) == 1:
            secs = '0'+secs
        return [mins, secs]

    @staticmethod
    def create_directory(directory):
        if not os.path.isdir(directory):
            os.mkdir(directory)

    @staticmethod
    def calculate_rmse(actual, targets):
        return np.sqrt((np.square(np.subtract(np.absolute(actual), np.absolute(targets)))).mean())

    @staticmethod
    def multinomial_likelihood(neural_network, data, weights):
        y = data[:, neural_network.topology[0]: neural_network.top[2]]
        fx = neural_network.generate_output(data, weights)
        rmse = self.calculate_rmse(fx, y) # Can be replaced by calculate_nmse function for reporting NMSE
        probability = neural_network.softmax(fx)
        loss = 0
        for index_1 in range(y.shape[0]):
            for index_2 in range(y.shape[1]):
                if y[index_1, index_2] == 1:
                    loss += np.log(probability[index_1, index_2])
        out = np.argmax(fx, axis=1)
        y_out = np.argmax(y, axis=1)
        count = 0
        for index in range(y_out.shape[0]):
            if out[index] == y_out[index]:
                count += 1
        accuracy = float(count)/y_out.shape[0] * 100
        return [loss, rmse, accuracy]

    @staticmethod
    def classification_prior(sigma_squared, weights):
        part_1 = -1 * ((weights.shape[0]) / 2) * np.log(2*np.p1*sigma_squared)
        part_2 = 1 / (2 * sigma_squared) * (sum(np.square(weights)))
        log_loss = part_1 - part_2
        return log_loss

    @staticmethod
    def gaussian_likelihood(neural_network, data, weights, tausq):
        desired = data[:, neural_network.topology[0]: neural_network.topology[0] + neural_network.topology[2]]
        prediction = neural_network.generate_output(data, weights)
        rmse = MCMC.calculate_rmse(prediction, desired)
        loss = -0.5 * np.log(2 * np.pi * tausq) - 0.5 * np.square(desired - prediction) / tausq
        return [np.sum(loss), rmse]

    @staticmethod
    def gaussian_prior(sigma_squared, nu_1, nu_2, weights, tausq):
        part1 = -1 * (weights.shape[0] / 2) * np.log(sigma_squared)
        part2 = 1 / (2 * sigma_squared) * (sum(np.square(weights)))
        log_loss = part1 - part2 - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
        return log_loss

    def likelihood_function(self, neural_network, data, weights, tau):
        if self.problem_type == 'regression':
            likelihood, rmse = self.gaussian_likelihood(neural_network, data, weights, tau)
        elif self.problem_type == 'classification':
            likelihood, rmse, accuracy = self.multinomial_likelihood(neural_network, data, weights)
        return likelihood, rmse

    def prior_function(self, weights, tau):
        if self.problem_type == 'regression':
            loss = self.gaussian_prior(self.sigma_squared, self.nu_1, self.nu_2, weights, tau)
        elif self.problem_type == 'classification':
            loss = self.classification_prior(self.sigma_squared, weights)
        return loss

    def evaluate_proposal(self, neural_network, train_data, test_data, weights_proposal, tau_proposal, likelihood_current, prior_current):
        accept = False
        likelihood_ignore, rmse_test_proposal = self.likelihood_function(neural_network, test_data, weights_proposal, tau_proposal)
        likelihood_proposal, rmse_train_proposal = self.likelihood_function(neural_network, train_data, weights_proposal, tau_proposal)
        prior_proposal = self.prior_function(weights_proposal, tau_proposal)
        difference_likelihood = likelihood_proposal - likelihood_current
        difference_prior = prior_proposal - prior_current
        mh_sum = difference_likelihood+difference_prior
        u = np.log(np.random.uniform(0,1))
        print(likelihood_proposal,prior_proposal)
        print(likelihood_current,prior_current)
        if u < mh_ratio :
            accept = True
            likelihood_current = likelihood_proposal
            prior_current = prior_proposal 
        return accept, rmse_train_proposal, rmse_test_proposal, likelihood_current, prior_current

    


    def mcmc_sampler(self, save_knowledge=True):           
        train_rmse_file = open(self.directory+'/train_rmse.csv', 'w')
        test_rmse_file = open(self.directory+'/test_rmse.csv', 'w')
        weights_initial = np.random.uniform(-5, 5, self.w_size)

        # ------------------- initialize MCMC-------------------------
        self.start_time = time.time()

        train_size = self.train_data.shape[0]
        test_size = self.test_data.shape[0]
        y_test = self.test_data[:, self.topology[0]: self.topology[0] + self.topology[2]]
        y_train = self.train_data[:, self.topology[0]: self.topology[0] + self.topology[2]]
        weights_current = weights_initial.copy()
        weights_proposal = weights_initial.copy()
        prediction_train = self.neural_network.generate_output(self.train_data, weights_current)
        prediction_test = self.neural_network.generate_output(self.test_data, weights_current)
        eta = np.log(np.var(prediction_train - y_train))
        tau_proposal = np.exp(eta)
        prior = self.prior_function(weights_current, tau_proposal)
        [likelihood, rmse_train] = self.likelihood_function(self.neural_network, self.train_data, weights_current, tau_proposal)
        
        rmse_test = self.calculate_rmse(prediction_test, y_test)

        # save values into previous variables
        rmse_train_current = rmse_train
        rmse_test_current = rmse_test
        num_accept = 0

        
        for sample in range(self.num_samples):
            self.swarm, self.best_swarm_pos, self.best_swarm_err=self.evolve(self.swarm, self.best_swarm_pos, self.best_swarm_err)
            weights_proposal = self.best_swarm_pos+np.random.normal(0, self.wpos_stepsize, self.w_size)
            eta_proposal = eta + np.random.normal(0, self.eta_stepsize, 1)
            tau_proposal = np.exp(eta_proposal)
            accept, rmse_train, rmse_test, likelihood, prior = self.evaluate_proposal(self.neural_network, self.train_data, self.test_data, weights_proposal, tau_proposal, likelihood, prior)
            
            if accept:
                num_accept += 1
                weights_current = weights_proposal
                self.best_swarm_pos=weights_current
                self.best_part_err=rmse_train
                eta = eta_proposal
                # save values into previous variables
                rmse_train_current = rmse_train
                rmse_test_current = rmse_test
                

            

            if save_knowledge:
                np.savetxt(train_rmse_file, [rmse_train_current])
                np.savetxt(test_rmse_file, [rmse_test_current])

            elapsed_time = ":".join(MCMC.convert_time(time.time() - self.start))
            #df_train=pd.read_csv()

            
            
            
            print("Sample: {}, Best Fitness Train: {}, Best Fitness Test: {},Proposal: {}, Time Elapsed: {},accept ratio: {}".format(sample, rmse_train_current,rmse_test_current,rmse_train, elapsed_time,num_accept/(sample+1)))
        burnin=0.1*(self.num_samples)
        #avg_rmse_train=np.mean(rmse_train[])
        #std_rmse_train=np.std(rmse_train)

        #print("Average RMSE train: {}, Train RMSE SD: {}, Train best: {}".format())
        elapsed_time = time.time() - self.start
        accept_ratio = num_accept/num_samples

        # Close the files
        train_rmse_file.close()
        test_rmse_file.close()

        return accept_ratio
    def mcmc_sampler_conventional(self, save_knowledge=True):           
        train_rmse_file = open(self.directory+'/train_rmse.csv', 'w')
        test_rmse_file = open(self.directory+'/test_rmse.csv', 'w')
        weights_initial = np.random.uniform(-5, 5, self.w_size)

        # ------------------- initialize MCMC-------------------------
        self.start_time = time.time()

        train_size = self.train_data.shape[0]
        test_size = self.test_data.shape[0]
        y_test = self.test_data[:, self.topology[0]: self.topology[0] + self.topology[2]]
        y_train = self.train_data[:, self.topology[0]: self.topology[0] + self.topology[2]]
        weights_current = weights_initial.copy()
        weights_proposal = weights_initial.copy()
        prediction_train = self.neural_network.generate_output(self.train_data, weights_current)
        prediction_test = self.neural_network.generate_output(self.test_data, weights_current)
        eta = np.log(np.var(prediction_train - y_train))
        tau_proposal = np.exp(eta)
        prior = self.prior_function(weights_current, tau_proposal)
        [likelihood, rmse_train] = self.likelihood_function(self.neural_network, self.train_data, weights_current, tau_proposal)
        
        rmse_test = self.calculate_rmse(prediction_test, y_test)

        # save values into previous variables
        rmse_train_current = rmse_train
        rmse_test_current = rmse_test
        num_accept = 0
        sample=0
        
        while(sample<self.num_samples):
            self.swarm, self.best_swarm_pos, self.best_swarm_err=self.evolve(self.swarm, self.best_swarm_pos, self.best_swarm_err)
            weights_proposal = self.best_swarm_pos+np.random.normal(0, self.wpos_stepsize, self.w_size)
            eta_proposal = eta + np.random.normal(0, self.eta_stepsize, 1)
            tau_proposal = np.exp(eta_proposal)
            accept, rmse_train, rmse_test, likelihood, prior = self.evaluate_proposal(self.neural_network, self.train_data, self.test_data, weights_proposal, tau_proposal, likelihood, prior)
            
            if accept:
                num_accept += 1
                weights_current = weights_proposal
                self.best_swarm_pos=weights_current
                self.best_part_err=rmse_train
                eta = eta_proposal
                # save values into previous variables
                rmse_train_current = rmse_train
                rmse_test_current = rmse_test
                

            

            if save_knowledge:
                np.savetxt(train_rmse_file, [rmse_train_current])
                np.savetxt(test_rmse_file, [rmse_test_current])

            elapsed_time = ":".join(MCMC.convert_time(time.time() - self.start))
            #df_train=pd.read_csv()

            

            
            
            print("Sample: {}, Best Fitness Train: {}, Best Fitness Test: {},Proposal: {}, Time Elapsed: {},accept ratio: {}".format(sample, rmse_train_current,rmse_test_current,rmse_train, elapsed_time,num_accept/(sample+1)))
            sample=sample+1
            for particle in range(self.pop_size):
                weights_proposal = self.swarm[particle].position+np.random.normal(0, self.wpos_stepsize, self.w_size)
                eta_proposal = eta + np.random.normal(0, self.eta_stepsize, 1)
                tau_proposal = np.exp(eta_proposal)
                accept, rmse_train, rmse_test, likelihood, prior = self.evaluate_proposal(self.neural_network, self.train_data, self.test_data, weights_proposal, tau_proposal, likelihood, prior)
            
                if accept:
                    num_accept += 1
                    weights_current = weights_proposal
                    self.swarm[particle].position=weights_current
                    #self.best_part_err=rmse_train
                    eta = eta_proposal
                    # save values into previous variables
                    rmse_train_current = rmse_train
                    rmse_test_current = rmse_test
                    

            

                if save_knowledge:
                    np.savetxt(train_rmse_file, [rmse_train_current])
                    np.savetxt(test_rmse_file, [rmse_test_current])

                elapsed_time = ":".join(MCMC.convert_time(time.time() - self.start))
                #df_train=pd.read_csv()

                

                
                
                print("Sample: {}, Best Fitness Train: {}, Best Fitness Test: {},Proposal: {}, Time Elapsed: {},accept ratio: {}".format(sample, rmse_train_current,rmse_test_current,rmse_train, elapsed_time,num_accept/(sample+1)))
                sample+=1


        burnin=0.1*(self.num_samples)
        #avg_rmse_train=np.mean(rmse_train[])
        #std_rmse_train=np.std(rmse_train)

        #print("Average RMSE train: {}, Train RMSE SD: {}, Train best: {}".format())
        elapsed_time = time.time() - self.start
        accept_ratio = num_accept/num_samples

        # Close the files
        train_rmse_file.close()
        test_rmse_file.close()

        return accept_ratio

if __name__ == '__main__':
    num_samples = 50000
    population_size = 50
    problem_type = 'regression'
    topology = [4, 5, 1]
    time_series_data=os.listdir("data_timeseries")
    for dataset in time_series_data:
    problem_name='LazerResults'
    dataset='Lazer' 
    train_data_file = 'data_timeseries/' + dataset+ '/train.txt'
    test_data_file = 'data_timeseries/' + dataset+ '/test.txt'
    train_data = np.genfromtxt(train_data_file, delimiter=' ',dtype=None)
    test_data = np.genfromtxt(test_data_file, delimiter=' ',dtype=None)
    model = MCMC(num_samples, population_size, topology, train_data, test_data, directory=problem_name)
    accept_ratio = model.mcmc_sampler()
    print("accept ratio: {}".format(accept_ratio))
