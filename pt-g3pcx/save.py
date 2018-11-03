def softmax(self, fx):
    ex = np.exp(fx)
    sum_ex = np.sum(ex, axis=1)
    sum_ex = np.multiply(np.ones(ex.shape), sum_ex[:, np.newaxis])
    prob = np.divide(ex, sum_ex)
    return prob

def rmse(self, predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def likelihood_func(self, neuralnet, data, w, tausq):
    y = data[:, self.topology[0]:]
    fx = neuralnet.generate_output(data, w)
    rmse = self.rmse(fx, y)
    prob = self.softmax(fx)
    # print prob.shape
    # loss = np.sum(-0.5 * np.log(2 * math.pi * tausq) - 0.5 * np.square(y - fx) / tausq)
    loss = 0
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if y[i, j] == 1:
                loss += np.log(prob[i, j] + 0.0001)

    out = np.argmax(fx, axis=1)
    y_out = np.argmax(y, axis=1)
    count = 0
    for i in range(y_out.shape[0]):
        if out[i] == y_out[i]:
            count += 1
    acc = float(count) / y_out.shape[0] * 100
    # print count
    # loss = np.log(np.sum(np.multiply(prob, y), axis=1))
    # print np.sum(loss)
    return [loss, fx, rmse, acc]

def prior_likelihood(self, sigma_squared, nu_1, nu_2, w, tausq):
    h = self.topology[1]  # number hidden neurons
    d = self.topology[0]  # number input neurons
    part1 = -1 * ((d * h + h + 2) / 2) * np.log(sigma_squared)
    part2 = 1 / (2 * sigma_squared) * (sum(np.square(w)))
    # log_loss = part1 - part2 - (1 + nu_1) * np.log(tausq) - (nu_2 / tausq)
    log_loss = part1 - part2
    return log_loss

def run(self):

    # ------------------- initialize MCMC

    # start = time.time()
    testsize = self.test_data.shape[0]
    trainsize = self.train_data.shape[0]
    samples = self.num_samples

    x_test = np.linspace(0, 1, num=testsize)
    x_train = np.linspace(0, 1, num=trainsize)

    train_acc = np.zeros((samples,))
    test_acc = np.zeros((samples,))

    netw = self.topology  # [input, hidden, output]
    y_test = self.test_data[:, netw[0]:]
    y_train = self.train_data[:, netw[0]:]
    # print y_train.size
    # print y_test.size

    w_size = (netw[0] * netw[1]) + (netw[1] * netw[2]) + netw[1] + netw[2]  # num of weights and bias

    pos_w = np.ones((samples, w_size))  # posterior of all weights and bias over all samples
    pos_tau = np.ones((samples, 1))

    fxtrain_samples = np.ones((samples, trainsize, self.topology[2]))  # fx of train data over all samples
    fxtest_samples = np.ones((samples, testsize, self.topology[2]))  # fx of test data over all samples
    rmse_train = np.zeros(samples)
    rmse_test = np.zeros(samples)

    w = np.random.randn(w_size)
    w_proposal = np.random.randn(w_size)

    step_w = 0.02;  # defines how much variation you need in changes to w
    step_eta = 0.01;
    # --------------------- Declare FNN and initialize

    neuralnet = Network(self.topology, self.train_data, self.test_data)
    # print 'evaluate Initial w'

    pred_train = neuralnet.generate_output(self.train_data, w)
    pred_test = neuralnet.generate_output(self.test_data, w)

    eta = np.log(np.var(pred_train - y_train))
    tau_pro = np.exp(eta)

    sigma_squared = 25
    nu_1 = 0
    nu_2 = 0

    prior = self.prior_likelihood(sigma_squared, nu_1, nu_2, w, tau_pro)  # takes care of the gradients

    [likelihood, pred_train, rmsetrain, trainacc] = self.likelihood_func(neuralnet, self.train_data, w, tau_pro)
    [likelihood_ignore, pred_test, rmsetest, testacc] = self.likelihood_func(neuralnet, self.test_data, w, tau_pro)

    train_acc[0] = trainacc
    test_acc[0] = testacc

    naccept = 0
    nreject = 0
    cnt = 0
    # print 'begin sampling using mcmc random walk'

    for i in range(samples - 1):

        w_proposal = w + np.random.normal(0, step_w, w_size)

        eta_pro = eta + np.random.normal(0, step_eta, 1)
        # tau_pro = math.exp(eta_pro)

        [likelihood_proposal, pred_train, rmsetrain, trainacc] = self.likelihood_func(neuralnet, self.train_data, w_proposal, tau_pro)
        [likelihood_ignore, pred_test, rmsetest, testacc] = self.likelihood_func(neuralnet, self.test_data, w_proposal, tau_pro)

        # likelihood_ignore  refers to parameter that will not be used in the alg.

        prior_prop = self.prior_likelihood(sigma_squared, nu_1, nu_2, w_proposal,
                                           tau_pro)  # takes care of the gradients

        diff_likelihood = likelihood_proposal - likelihood
        diff_prior = prior_prop - prior

        mh_prob = min(1, math.exp(diff_likelihood))

        u = random.uniform(0, 1)

        if u < mh_prob:
            # Update position
            # print    i, ' is accepted sample'
            naccept += 1
            likelihood = likelihood_proposal
            prior_likelihood = prior_prop
            w = w_proposal
            eta = eta_pro
            accept = True

            # print i, trainacc, rmsetrain

            # print  i, likelihood, prior_likelihood, rmsetrain, rmsetest, 'accepted: ', naccept , trainacc, testacc
            # print pred_train.shape
            # print fxtrain_samples
            # print(likelihood)
            # print 'accepted'

            pos_w[i + 1,] = w_proposal
            pos_tau[i + 1,] = tau_pro
            fxtrain_samples[i + 1,] = pred_train
            fxtest_samples[i + 1,] = pred_test
            rmse_train[i + 1,] = rmsetrain
            rmse_test[i + 1,] = rmsetest
            train_acc[i + 1] = trainacc
            # plt.plot(x_train, pred_train)
            test_acc[i + 1] = testacc


        else:
            pos_w[i + 1,] = pos_w[i,]
            pos_tau[i + 1,] = pos_tau[i,]
            fxtrain_samples[i + 1,] = fxtrain_samples[i,]
            fxtest_samples[i + 1,] = fxtest_samples[i,]
            rmse_train[i + 1,] = rmse_train[i,]
            rmse_test[i + 1,] = rmse_test[i,]
            train_acc[i + 1] = train_acc[i]
            test_acc[i + 1] = test_acc[i]
            accept = False
            if rmse_train[i] > rmsetrain:
                print i, rmse_train[i], rmsetrain
                cnt += 1
            nreject += 1

        # print "{:.2f}, {:.2f}".format(mh_prob, u)
        # time.sleep(0.1)
            # print i, 'rejected and retained'
        # elapsed = convert_time(time.time() - start)
        # sys.stdout.write('\rSamples: ' + str(i + 2) + "/" + str(samples) + " Time elapsed: " + str(elapsed[0]) + ":" + str(elapsed[1]))

        # print naccept, ' num accepted'
    print cnt, nreject, samples
    print naccept / float(samples) * 100.0, '% was accepted'
    accept_ratio = naccept / (samples * 1.0) * 100

    return (
    pos_w, pos_tau, fxtrain_samples, fxtest_samples, x_train, x_test, rmse_train, rmse_test, train_acc, test_acc,
    accept_ratio)
