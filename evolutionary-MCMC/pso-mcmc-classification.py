
import numpy as np
import random
import time
import math
import pandas as pd
import os
from pso import PSO, Particle

class Network:

    def __init__(self, topology, train_data, test_data, learn_rate = 0.5, alpha = 0.1):
        self.topology = topology  # NN topology [input, hidden, output]
        np.random.seed(int(time.time()))
        self.train_data = train_data
        self.test_data = test_data
        self.W1 = np.random.randn(self.topology[0], self.topology[1]) / np.sqrt(self.topology[0]),
        self.B1 = np.random.randn(1, self.topology[1]) / np.sqrt(self.topology[1])  # bias first layer
        self.W2 = np.random.randn(self.topology[1], self.topology[2]) / np.sqrt(self.topology[1])
        self.B2 = np.random.randn(1, self.topology[2]) / np.sqrt(self.topology[1])  # bias second layer
        self.hidout = np.zeros((1, self.topology[1]))  # output of first hidden layer
        self.out = np.zeros((1, self.topology[2]))  # output last layer
        self.pred_class=0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sampleEr(self, actualout):
        error = np.subtract(self.out, actualout)
        sqerror = np.sum(np.square(error)) / self.topology[2]
        return sqerror

    def ForwardPass(self, X):
        z1 = X.dot(self.W1) - self.B1
        self.hidout = self.sigmoid(z1)  # output of first hidden layer
        z2 = self.hidout.dot(self.W2) - self.B2
        self.out = self.sigmoid(z2)  # output second hidden layer

        self.pred_class = np.argmax(self.out)


        #print(self.pred_class, self.out, '  ---------------- out ')

    '''def BackwardPass(self, Input, desired):
        out_delta = (desired - self.out).dot(self.out.dot(1 - self.out))
        hid_delta = out_delta.dot(self.W2.T) * (self.hidout * (1 - self.hidout))
        print(self.B2.shape)
        self.W2 += (self.hidout.T.reshape(self.Top[1],1).dot(out_delta) * self.lrate)
        self.B2 += (-1 * self.lrate * out_delta)
        self.W1 += (Input.T.reshape(self.Top[0],1).dot(hid_delta) * self.lrate)
        self.B1 += (-1 * self.lrate * hid_delta)'''


    def calculate_rmse(self,predict, targets):
        #targets=np.argmax(targets,axis=1)
        #print(predict,targets)
        return np.sqrt((np.square(np.subtract(np.absolute(predict), np.absolute(targets)))).mean())
    
    def calculate_mse(self,predict, targets):
        #targets=np.argmax(targets,axis=1)
        #print(predict,targets)
        return ((np.square(np.subtract(np.absolute(predict), np.absolute(targets)))).mean())
    
    

    def decode(self, w):
        w_layer1size = self.topology[0] * self.topology[1] 
        w_layer2size = self.topology[1] * self.topology[2]

        w_layer1 = w[0:w_layer1size]
        self.W1 = np.reshape(w_layer1, (self.topology[0], self.topology[1]))

        w_layer2 = w[w_layer1size:w_layer1size + w_layer2size]
        self.W2 = np.reshape(w_layer2, (self.topology[1], self.topology[2]))
        self.B1 = w[w_layer1size + w_layer2size:w_layer1size + w_layer2size + self.topology[1]].reshape(1,self.topology[1])
        self.B2 = w[w_layer1size + w_layer2size + self.topology[1]:w_layer1size + w_layer2size + self.topology[1] + self.topology[2]].reshape(1,self.topology[2])

 

    def encode(self):
        w1 = self.W1.ravel()
        w1 = w1.reshape(1,w1.shape[0])
        w2 = self.W2.ravel()
        w2 = w2.reshape(1,w2.shape[0])
        w = np.concatenate([w1.T, w2.T, self.B1.T, self.B2.T])
        w = w.reshape(-1)
        return w

    @staticmethod
    def softmax(fx):
        ex = np.exp(fx)
        sum_ex = np.sum(ex,axis=1)
        sum_ex = np.multiply(np.ones(ex.shape), sum_ex[:, np.newaxis])
        probability = np.divide(ex, sum_ex)
        return probability
 


    '''def langevin_gradient(self, data, w, depth):  # BP with SGD (Stocastic BP)

        self.decode(w)  # method to decode w into W1, W2, B1, B2.
        size = data.shape[0]

        Input = np.zeros((1, self.Top[0]))  # temp hold input
        Desired = np.zeros((1, self.Top[2]))
        fx = np.zeros(size)

        for i in range(0, depth):
            for i in range(0, size):
                pat = i
                Input = data[pat, 0:self.Top[0]]
                Desired = data[pat, self.Top[0]:]
                self.ForwardPass(Input)
                self.BackwardPass(Input, Desired)
        w_updated = self.encode()

        return  w_updated'''

    def generate_output(self, data, w ):  # BP with SGD (Stocastic BP)

        self.decode(w)  # method to decode w into W1, W2, B1, B2.
        size = data.shape[0]

        Input = np.zeros((1, self.topology[0]))  # temp hold input
        Desired = np.zeros((1, self.topology[2]))
        fx = np.zeros((size,self.topology[2]))
        prob = np.zeros((size,self.topology[2]))

        for i in range(0, size):  # to see what fx is produced by your current weight update
            Input = data[i, 0:self.topology[0]]
            self.ForwardPass(Input)
            fx[i] = self.out
            
        prob=self.softmax(fx)
        #print(fx, 'fx')
        #print(prob, 'prob' )

        return fx, prob

    def accuracy(self , pred, actual):
        count=0 
        actual=np.argmax(actual,axis=1)
        prob=self.softmax(pred)
        prob=np.argmax(prob,axis=1)
        for i in range(prob.shape[0]):
            if prob[i] == actual[i]:
                count+=1
        return (float(count)/pred.shape[0])*100

        

    def classification_perf(self, x, type_data):

        if type_data == 'train':
            data = self.train_data
        else:
            data = self.test_data

        y = (data[:, self.topology[0]:self.topology[0] + self.topology[2]])
        #print(np.shape(y))
        fx, prob = self.generate_output(data,x)
        fit= self.calculate_rmse(fx,y) 
        acc = self.accuracy(fx,y) 

        return acc, fit

        
    def evaluate_fitness(self, x):    #  function  (can be any other function, model or diff neural network models (FNN or RNN ))
          
        acc,fit=self.classification_perf(x,'train')



        return fit  #fit # note we will maximize fitness, hence minimize error

class MCMC(PSO, Particle):
    def __init__(self, num_samples, population_size, topology, train_data, test_data, directory, problem_type='classification', max_limit=2, min_limit=-2):
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


    

    def initialize_sampling_parameters(self):
        self.eta_stepsize = 0.25

        self.wpos_stepsize=0.005
        self.sigma_squared = 50
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
        y = (data[:, neural_network.topology[0]:neural_network.topology[0]+ neural_network.topology[2]])
        #print((y[0]))
        fx,probability = neural_network.generate_output(data, weights)
        mse = neural_network.calculate_rmse(fx, y) # Can be replaced by calculate_nmse function for reporting NMSE
        #probability = neural_network.softmax(fx)
        #print(probability[6])
        loss = 0
        for index_1 in range(y.shape[0]):
            for index_2 in range(y.shape[1]):
                if y[index_1, index_2] == 1:
                    loss += np.log(probability[index_1, index_2])
        out = np.argmax(probability, axis=1)
        y_out = np.argmax(y, axis=1)
        count = 0
        for index in range(y_out.shape[0]):
            if out[index] == y_out[index]:
                count += 1
        accuracy = (count)/y_out.shape[0] * 100
        #print(rmse)
        return [loss, mse, accuracy]

    @staticmethod
    def classification_prior(sigma_squared, weights):
        part_1 = -1 * ((weights.shape[0]) / 2) * np.log(2*np.pi*sigma_squared)
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
        return likelihood, rmse,accuracy

    def prior_function(self, weights, tau):
        if self.problem_type == 'regression':
            loss = self.gaussian_prior(self.sigma_squared, self.nu_1, self.nu_2, weights, tau)
        elif self.problem_type == 'classification':
            loss = self.classification_prior(self.sigma_squared, weights)
        return loss

    def evaluate_proposal(self, neural_network, train_data, test_data, weights_proposal, tau_proposal, likelihood_current, prior_current):
        accept = False
        likelihood_ignore, rmse_test_proposal,accuracy_test_proposal = self.likelihood_function(neural_network, test_data, weights_proposal, tau_proposal)
        #print(rmse_test_proposal,accuracy_test_proposal)
        likelihood_proposal, rmse_train_proposal,accuracy_train_proposal = self.likelihood_function(neural_network, train_data, weights_proposal, tau_proposal)
        #print(np.shape(rmse_test_proposal))
        prior_proposal = self.prior_function(weights_proposal, tau_proposal)
        difference_likelihood = likelihood_proposal - likelihood_current
        difference_prior = prior_proposal - prior_current
        print(likelihood_proposal,likelihood_current)
        print(prior_proposal,prior_current)
        mh_ratio = difference_likelihood+difference_prior
        #mh_ratio=0.5
        u = np.log(np.random.uniform(0,1))
        #print(mh_ratio) 
        if u < mh_ratio :
            accept = True
            likelihood_current = likelihood_proposal
            prior_current = prior_proposal 

        return accept, rmse_train_proposal, rmse_test_proposal,accuracy_train_proposal,accuracy_test_proposal, likelihood_current, prior_current



    def mcmc_sampler(self, save_knowledge=True):           
        train_rmse_file = open(self.directory+'/train_rmse.csv', 'w')
        test_rmse_file = open(self.directory+'/test_rmse.csv', 'w')
        #accept_ratio_file = open(self.directory+'/ar.txt', 'w')
        train_accuracy_file = open(self.directory+'/train_acuracy.csv', 'w')
        test_accuracy_file = open(self.directory+'/test_accuracy.csv', 'w')
        weights_initial = np.random.uniform(-0.2, 0.2, self.w_size)

        # ------------------- initi
        # alize MCMC-------------------------
        self.start_time = time.time()

        train_size = self.train_data.shape[0]
        test_size = self.test_data.shape[0]
        y_test = self.test_data[:, self.topology[0]: self.topology[0] + self.topology[2]]
        y_train = self.train_data[:, self.topology[0]: self.topology[0] + self.topology[2]]
        weights_current = weights_initial.copy()
        weights_proposal = weights_initial.copy()
        prediction_train,probability_train = self.neural_network.generate_output(self.train_data, weights_current)
        prediction_test,probability_test = self.neural_network.generate_output(self.test_data, weights_current)
        eta = 0
        tau_proposal = 1 
        prior = self.prior_function(weights_current, tau_proposal)
        [likelihood, rmse_train,accuracy_train] = self.likelihood_function(self.neural_network, self.train_data, weights_current, tau_proposal)
        accuracy_test,rmse_test=self.neural_network.classification_perf(weights_current, 'test')
        


        # save values into previous variables
        rmse_train_current = rmse_train
        rmse_test_current = rmse_test
        accuracy_train_current = accuracy_train
        accuracy_test_current = accuracy_test
        num_accept = 0

        
        for sample in range(self.num_samples):
            
            self.swarm, self.best_swarm_pos, self.best_swarm_err=self.evolve(self.swarm, self.best_swarm_pos, self.best_swarm_err)
            #print( self.best_swarm_pos)
            weights_proposal =  self.best_swarm_pos+  np.random.normal(0, self.wpos_stepsize, self.w_size)
            print(weights_proposal)

            print(weights_proposal)
            #print(weights_proposal)
            eta_proposal = eta + np.random.normal(0, self.eta_stepsize, 1)
            tau_proposal = np.exp(eta_proposal)
            accept, rmse_train, rmse_test,accuracy_train, accuracy_test, likelihood, prior = self.evaluate_proposal(self.neural_network, self.train_data, self.test_data, weights_proposal, tau_proposal, likelihood, prior)
            #print(accuracy_train, accuracy_test)
            
            if accept:
                num_accept += 1
                weights_current = weights_proposal
                #print(weights_current)
                self.best_swarm_pos=weights_proposal
                self.best_part_err=rmse_train
                #eta = eta_proposal
                # save values into previous variables
                rmse_train_current = rmse_train
                rmse_test_current = rmse_test
                accuracy_train_current = accuracy_train
                accuracy_test_current = accuracy_test
                

            
              
            if save_knowledge:
                np.savetxt(train_rmse_file, [rmse_train_current])
                np.savetxt(test_rmse_file, [rmse_test_current])
                np.savetxt(train_accuracy_file, [accuracy_train_current])
                np.savetxt(test_accuracy_file, [accuracy_test_current])

            elapsed_time = ":".join(MCMC.convert_time(time.time() - self.start))
            #df_train=pd.read_csv()

            
            
            
            print("Sample: {}, Best Accuracy Train: {}, Best Accuracy Test: {},Proposal: {}, Time Elapsed: {},accept ratio: {}".format(sample, accuracy_train_current,accuracy_test_current,accuracy_train, elapsed_time,num_accept/(sample+1)))
        burnin=0.1*(self.num_samples)
        #avg_rmse_train=np.mean(rmse_train[])
        #std_rmse_train=np.std(rmse_train)

        #print("Average RMSE train: {}, Train RMSE SD: {}, Train best: {}".format())
        elapsed_time = time.time() - self.start
        
        accept_ratio = num_accept/num_samples

        # Close the files
        train_rmse_file.close()
        test_rmse_file.close()
        train_accuracy_file.close()
        test_accuracy_file.close()


        return accept_ratio

    def mcmc_sampler_conventional(self, save_knowledge=True):           
        train_rmse_file = open(self.directory+'/train_rmse.csv', 'w')
        test_rmse_file = open(self.directory+'/test_rmse.csv', 'w')
        train_accuracy_file = open(self.directory+'/train_acuracy.csv', 'w')
        test_accuracy_file = open(self.directory+'/test_accuracy.csv', 'w')
        accept_ratio_file = open(self.directory+'/ar.csv', 'w')
        weights_initial = np.random.uniform(-0.2, 0.2, self.w_size)

        # ------------------- initi
        # alize MCMC-------------------------
        self.start_time = time.time()

        train_size = self.train_data.shape[0]
        test_size = self.test_data.shape[0]
        y_test = self.test_data[:, self.topology[0]: self.topology[0] + self.topology[2]]
        y_train = self.train_data[:, self.topology[0]: self.topology[0] + self.topology[2]]
        weights_current = weights_initial.copy()
        weights_proposal = weights_initial.copy()
        prediction_train,probability_train = self.neural_network.generate_output(self.train_data, weights_current)
        prediction_test,probability_test = self.neural_network.generate_output(self.test_data, weights_current)
        eta = 0
        tau_proposal = 1 
        prior = self.prior_function(weights_current, tau_proposal)
        [likelihood, rmse_train,accuracy_train] = self.likelihood_function(self.neural_network, self.train_data, weights_current, tau_proposal)
        accuracy_test,rmse_test=self.neural_network.classification_perf(weights_current, 'test')
        


        # save values into previous variables
        rmse_train_current = rmse_train
        rmse_test_current = rmse_test
        accuracy_train_current = accuracy_train
        accuracy_test_current = accuracy_test
        num_accept = 0
        sample=0
        
        while(sample<self.num_samples):
            
            self.swarm, self.best_swarm_pos, self.best_swarm_err=self.evolve(self.swarm, self.best_swarm_pos, self.best_swarm_err)

            weights_proposal =  self.best_swarm_pos+  np.random.normal(0, self.wpos_stepsize, self.w_size)
            #print(weights_proposal)
            eta_proposal = eta + np.random.normal(0, self.eta_stepsize, 1)
            tau_proposal = np.exp(eta_proposal)
            accept, rmse_train, rmse_test,accuracy_train, accuracy_test, likelihood, prior = self.evaluate_proposal(self.neural_network, self.train_data, self.test_data, weights_proposal, tau_proposal, likelihood, prior)
            #print(accuracy_train, accuracy_test)
            
            if accept:
                num_accept += 1
                weights_current = weights_proposal
                #print(weights_current)
                self.best_swarm_pos=weights_proposal
                self.best_part_err=rmse_train
                #eta = eta_proposal
                # save values into previous variables
                rmse_train_current = rmse_train
                rmse_test_current = rmse_test
                accuracy_train_current = accuracy_train
                accuracy_test_current = accuracy_test
                

            
              
            if save_knowledge:
                np.savetxt(train_rmse_file, [rmse_train_current])
                np.savetxt(test_rmse_file, [rmse_test_current])
                np.savetxt(train_accuracy_file, [accuracy_train_current])
                np.savetxt(test_accuracy_file, [accuracy_test_current])

            for particle in range(self.pop_size):

                #print( self.best_swarm_pos)
                weights_proposal =  self.swarm[particle].position +  np.random.normal(0, self.wpos_stepsize, self.w_size)
                #print(weights_proposal)

                #print(weights_proposal)
                #print(weights_proposal)
                eta_proposal = eta + np.random.normal(0, self.eta_stepsize, 1)
                tau_proposal = np.exp(eta_proposal)
                accept, rmse_train, rmse_test,accuracy_train, accuracy_test, likelihood, prior = self.evaluate_proposal(self.neural_network, self.train_data, self.test_data, weights_proposal, tau_proposal, likelihood, prior)
                #print(accuracy_train, accuracy_test)
                
                if accept:
                    num_accept += 1
                    weights_current = weights_proposal
                    #print(weights_current)
                    self.swarm[particle].position=weights_proposal
                    
                    #eta = eta_proposal
                    # save values into previous variables
                    rmse_train_current = rmse_train
                    rmse_test_current = rmse_test
                    accuracy_train_current = accuracy_train
                    accuracy_test_current = accuracy_test
                    

                
                
                if save_knowledge:
                    np.savetxt(train_rmse_file, [rmse_train_current])
                    np.savetxt(test_rmse_file, [rmse_test_current])
                    np.savetxt(train_accuracy_file, [accuracy_train_current])
                    np.savetxt(test_accuracy_file, [accuracy_test_current])

                elapsed_time = ":".join(MCMC.convert_time(time.time() - self.start))
                #df_train=pd.read_csv()

                
                
                
                print("Sample: {}, Best Accuracy Train: {}, Best Accuracy Test: {},Proposal: {}, Time Elapsed: {},accept ratio: {}".format(sample, accuracy_train_current,accuracy_test_current,accuracy_train, elapsed_time,num_accept/((sample+1))))
                sample+=1
        burnin=0.1*(self.num_samples)
        #avg_rmse_train=np.mean(rmse_train[])
        #std_rmse_train=np.std(rmse_train)

        #print("Average RMSE train: {}, Train RMSE SD: {}, Train best: {}".format())
        elapsed_time = time.time() - self.start
        
        accept_ratio = num_accept/num_samples
        np.savetxt(accept_ratio_file, [accept_ratio])
        # Close the files
        train_rmse_file.close()
        accept_ratio_file.close()
        test_rmse_file.close()
        train_accuracy_file.close()
        test_accuracy_file.close()

        return accept_ratio

if __name__ == '__main__':
    num_samples = 50000
    population_size = 50
    problem_type = 'classification'
    topology = [8,12,2]
    #time_series_data=os.listdir("data_classification")
    #for dataset in time_series_data:
    problem_name='CancerResults'
    train_data_file = 'data_classification/Cancer/ftrain.txt'
    test_data_file = 'data_classification/Cancer/ftest.txt'
    train_data = np.genfromtxt(train_data_file,delimiter=',',dtype=None)
    test_data = np.genfromtxt(test_data_file,delimiter=',',dtype=None)
    #train_data=pd.read_csv(train_data_file).values
    #test_data=pd.read_csv(test_data_file).values
    
    model = MCMC(num_samples, population_size, topology, train_data, test_data, directory=problem_name)
    accept_ratio = model.mcmc_sampler_conventional()
    print("accept ratio: {}".format(accept_ratio))
