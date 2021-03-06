import numpy as np
import tensorflow as tf
import gym
import logz
import scipy.signal

import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 


tiny = 1e-10

def normc_initializer(std=1.0):
    """
    Initialize array with normalized columns
    """
    def _initializer(shape, dtype=None, partition_info=None): #pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def dense(x, size, name, weight_init=None):
    """
    Dense (fully connected) layer
    """
    if isinstance(weight_init,float):
        weight_init = normc_initializer(weight_init)
    if weight_init is None:
        weight_init = tf.contrib.layers.xavier_initializer()
    with tf.variable_scope(name):
        w = tf.get_variable('weight', [x.get_shape()[1], size], initializer=weight_init)
        b = tf.get_variable('bias', [size], initializer=tf.constant_initializer(0, dtype=tf.float32))
    return tf.matmul(x, w) + b

def fancy_slice_2d(X, inds0, inds1):
    """
    Like numpy's X[inds0, inds1]
    """
    inds0 = tf.cast(inds0, tf.int64)
    inds1 = tf.cast(inds1, tf.int64)
    shape = tf.cast(tf.shape(X), tf.int64)
    ncols = shape[1]
    Xflat = tf.reshape(X, [-1])
    return tf.gather(Xflat, inds0 * ncols + inds1)

def discount(x, gamma):
    """
    Compute discounted sum of future values
    out[i] = in[i] + gamma * in[i+1] + gamma^2 * in[i+2] + ...
    """
    return scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]

def explained_variance_1d(ypred,y):
    """
    Var[ypred - y] / var[y]. 
    https://www.quora.com/What-is-the-meaning-proportion-of-variance-explained-in-linear-regression
    """
    assert y.ndim == 1 and ypred.ndim == 1    
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary

def categorical_sample_logits(logits):
    """
    Samples (symbolically) from categorical distribution, where logits is a NxK
    matrix specifying N categorical distributions with K categories
    specifically, exp(logits) / sum( exp(logits), axis=1 ) is the 
    probabilities of the different classes
    Cleverly uses gumbell trick, based on
    https://github.com/tensorflow/tensorflow/issues/456
    """
    U = tf.random_uniform(tf.shape(logits))
    return tf.argmax(logits - tf.log(-tf.log(U)), dimension=1)

def pathlength(path):
    return len(path["reward"])

class LinearValueFunction(object):
    coef = None
    def fit(self, X, y):
        Xp = self.preproc(X)
        A = Xp.T.dot(Xp)
        nfeats = Xp.shape[1]
        A[np.arange(nfeats), np.arange(nfeats)] += 1e-3 # a little ridge regression
        b = Xp.T.dot(y)
        self.coef = np.linalg.solve(A, b)
    def predict(self, X):
        if self.coef is None:
            return np.zeros(X.shape[0])
        else:
            return self.preproc(X).dot(self.coef)
    def preproc(self, X):
        return np.concatenate([np.ones([X.shape[0], 1]), X, np.square(X)/2.0], axis=1)

class NnValueFunction(object):
    coeffs = None

    def __init__(self, session):
        self.net = None
        self.session = session

    def create_net(self, shape):
        self.x = tf.placeholder(tf.float32, shape=[None, shape], name="x")
        self.y = tf.placeholder(tf.float32, shape=[None], name="y")
        hidden1 = tf.nn.relu(dense(self.x, 32, 'value-net-hidden1', 1.0))
        hidden2 = tf.nn.relu(dense(hidden1, 16, 'value-net-hidden2', 1.0))
        self.net = dense(hidden2, 1, 'value-net-out', 1.0)
        self.net = tf.reshape(self.net, (-1,))
        l2 = (self.net - self.y) * (self.net - self.y)
        self.train = tf.train.AdamOptimizer().minimize(l2)
        self.session.run(tf.initialize_all_variables())

    def preproc(self, X):
        return np.concatenate([np.ones([X.shape[0], 1]), X, np.square(X)/2.0], axis=1)

    def fit(self, X, y):
        featmat = self.preproc(X)
        if self.net is None:
            self.create_net(featmat.shape[1])
        for _ in range(40):
            self.session.run(self.train, {self.x: featmat, self.y: y})

    def predict(self, X):
        if self.net is None:
            return np.zeros(X.shape[0])
        else:
            ret = self.session.run(self.net, {self.x: self.preproc(X)})
            return np.reshape(ret, (ret.shape[0],))


def lrelu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)

def normal_log_prob(x, mean, log_std, dim):
    """
    x: [batch, dim]
    return: [batch]
    """
    zs = (x - mean) / tf.exp(log_std)
    return - tf.reduce_sum(log_std, axis=1) - \
           0.5 * tf.reduce_sum(tf.square(zs), axis=1) - \
           0.5 * dim * np.log(2 * np.pi)


def normal_kl(old_mean, old_log_std, new_mean, new_log_std):
    """
    mean, log_std: [batch,  dim]
    return: [batch]
    """
    old_std = tf.exp(old_log_std)
    new_std = tf.exp(new_log_std)
    numerator = tf.square(old_mean - new_mean) + \
                tf.square(old_std) - tf.square(new_std)
    denominator = 2 * tf.square(new_std) + tiny
    return tf.reduce_sum(
        numerator / denominator + new_log_std - old_log_std, axis=1)


def normal_entropy(log_std):
    return tf.reduce_sum(log_std + np.log(np.sqrt(2 * np.pi * np.e)), axis=1)

def main_cartpole(n_iter=100, gamma=1.0, min_timesteps_per_batch=1000, stepsize=1e-2, animate=True, logdir=None, vf_type='linear'):
    tf.reset_default_graph()
    env = gym.make("CartPole-v0")
    ob_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n
    logz.configure_output_dir(logdir)

    # Symbolic variables have the prefix sy_, to distinguish them from the numerical values
    # that are computed later in these function
    sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32) # batch of observations
    sy_ac_n = tf.placeholder(shape=[None], name="ac", dtype=tf.int32) # batch of actions taken by the policy, used for policy gradient computation
    sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32) # advantage function estimate

    variableInit =  tf.Variable(initial_value = tf.zeros([10,16], tf.float32), validate_shape = False)

    with tf.variable_scope("SL_vars"):
        sy_h1 = lrelu(dense(sy_ob_no, 32, "h1", weight_init=normc_initializer(1.0))) # hidden layer
        sy_h2 = lrelu(dense(sy_h1, 32, "h2", weight_init=normc_initializer(1.0))) # hidden layer
        critical_layer = lrelu(dense(sy_h2, 16, "criticalLayer", weight_init=normc_initializer(1.0))) # hidden layer

    with tf.variable_scope("PG_vars"):
        #The weights of the two following layers are in the PG scope
        sy_h3 = lrelu(dense(critical_layer, 32, "h3", weight_init=normc_initializer(1.0))) # hidden layen
        sy_logits_na = dense(sy_h3, num_actions, "final", weight_init=normc_initializer(0.05))

    with tf.variable_scope("PG_vars", reuse = True):

        activationOp = tf.assign(variableInit, critical_layer, False)
        h3_back = lrelu(dense(activationOp, 32, "h3", weight_init=normc_initializer(1.0))) # hidden layer
        sy_logits_back = dense(h3_back, num_actions, "final", weight_init=normc_initializer(0.05))
    
    
    
    sy_sampled_ac = categorical_sample_logits(sy_logits_na)[0] # sampled actions, used for defining the policy (NOT computing the policy gradient)
    

    sy_logp = tf.nn.log_softmax(sy_logits_back) # logprobability of actions
    sy_n = tf.shape(sy_ob_no)[0]
    sy_logprob_n = fancy_slice_2d(sy_logp, tf.range(sy_n), sy_ac_n) # log-prob of actions taken -- used for policy gradient calculation

    sy_surr = - tf.reduce_mean(sy_adv_n * sy_logprob_n) # Loss function that we'll differentiate to get the policy gradient ("surr" is for "surrogate loss")
    l2_loss = tf.nn.l2_loss(tf.subtract(activationOp, critical_layer))
        #activationOp will change during the PG_step


    _PGvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='PG_vars')
    _SLvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='SL_vars')

        #sy_stepsize = tf.placeholder(shape=[], dtype=tf.float32) # Symbolic, in case you want to change the stepsize during optimization. (We're not doing that currently)
        #tf.global_variables_initializer().run() #pylint: disable=E1101
          
    
    optimizerPG = tf.train.AdamOptimizer()
    PG_step = optimizerPG.minimize(loss = sy_surr, var_list = _PGvars)

       
    optimizerSL = tf.train.AdamOptimizer(1e-4)
    SL_step = optimizerSL.minimize(loss =l2_loss, var_list = _SLvars)



    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1) 
    # use single thread. on such a small problem, multithreading gives you a slowdown
    # this way, we can better use multiple cores for different experiments
    #sess = tf.Session(config=tf_config)
    #sess.__enter__() # equivalent to `with sess:`

    sess = tf.InteractiveSession(config = tf_config)
    globalOp = tf.global_variables_initializer()
    sess.run(globalOp)

    
    if vf_type == 'linear':
        vf = LinearValueFunction()
    elif vf_type == 'nn':
        vf = NnValueFunction(sess)
    else:
        raise NotImplementedError

    total_timesteps = 0
    num_trajectories = 1000
    MeanRewardHistory = []
    for i in range(num_trajectories):
        print("********** Iteration %i ************"%i)

        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        while True:
            ob = env.reset()
            terminated = False
            obs, acs, rewards = [], [], []
            animate_this_episode=(len(paths)==0 and (i % 10 == 0) and animate)
            while True:
                if animate_this_episode:
                    env.render()
                obs.append(ob)
                ac = sess.run(sy_sampled_ac, feed_dict={sy_ob_no : ob[None]})
                acs.append(ac)
                ob, rew, done, _ = env.step(ac)
                rewards.append(rew)
                if done:
                    break                    
            path = {"observation" : np.array(obs), "terminated" : terminated,
                    "reward" : np.array(rewards), "action" : np.array(acs)}
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch
        # Estimate advantage function
        vtargs, vpreds, advs = [], [], []
        for path in paths:
            rew_t = path["reward"]
            return_t = discount(rew_t, gamma)
            vpred_t = vf.predict(path["observation"])
            adv_t = return_t - vpred_t
            advs.append(adv_t)
            vtargs.append(return_t)
            vpreds.append(vpred_t)

        # Build arrays for policy update
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_n = np.concatenate([path["action"] for path in paths])
        adv_n = np.concatenate(advs)
        standardized_adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)
        vtarg_n = np.concatenate(vtargs)
        vpred_n = np.concatenate(vpreds)
        vf.fit(ob_no, vtarg_n)


        sess.run(activationOp, feed_dict = {sy_ob_no:ob_no})
        sess.run(PG_step, feed_dict={sy_ob_no:ob_no, sy_ac_n:ac_n, sy_adv_n:standardized_adv_n} )
        sess.run(SL_step, feed_dict={sy_ob_no:ob_no} )
       
       
        # Log diagnostics
        EpRewMean = np.mean([path["reward"].sum() for path in paths])
        MeanRewardHistory.append(EpRewMean)
        logz.log_tabular("EpRewMean", EpRewMean)
        logz.log_tabular("EpLenMean", np.mean([pathlength(path) for path in paths]))
        #logz.log_tabular("KLOldNew", kl)
        #logz.log_tabular("Entropy", ent)
        logz.log_tabular("EVBefore", explained_variance_1d(vpred_n, vtarg_n))
        logz.log_tabular("EVAfter", explained_variance_1d(vf.predict(ob_no), vtarg_n))
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        # If you're overfitting, EVAfter will be way larger than EVBefore.
        # Note that we fit value function AFTER using it to compute the advantage function to avoid introducing bias
        logz.dump_tabular()

    Iterations= range(0,num_trajectories)
    plt.plot(Iterations, MeanRewardHistory, '-r')
    plt.plot(Iterations, [0]*num_trajectories, '-b')
    plt.xlabel("Iteration")
    plt.ylabel("Sum of Rewards")
    plt.axis([0, num_trajectories,0, 200])
    plt.title('Split Training')
    plt.savefig('SplitPlot_LinearBaseline.png')
        
        



def run(case):
    if case == 0 or case < 0:
        main_cartpole(logdir=None,vf_type='linear', animate=False) # when you want to start collecting results, set the logdir
    if case == 1 or case < 0:
        main_cartpole(logdir='./log/cartpole-nn',vf_type='nn',animate=False) # when you want to start collecting results, set the logdir    
    if case == 2 or case < 0:
        general_params = dict(gamma=0.97, animate=False, min_timesteps_per_batch=2500, n_iter=300, initial_stepsize=1e-3)
        params = [
            dict(logdir='./log/linearvf-kl2e-3-seed0', seed=0, desired_kl=2e-3, vf_type='linear', n_iter=500000, vf_params={}, **general_params),
            dict(logdir='./log/nnvf-kl2e-3-seed0', seed=0, desired_kl=2e-3, vf_type='nn', n_iter=300000, vf_params=dict(n_epochs=10, stepsize=1e-3), **general_params),
            dict(logdir='./log/linearvf-kl2e-3-seed1', seed=1, desired_kl=2e-3, vf_type='linear', n_iter=500000, vf_params={}, **general_params),
            dict(logdir='./log/nnvf-kl2e-3-seed1', seed=1, desired_kl=2e-3, vf_type='nn', n_iter=300000, vf_params=dict(n_epochs=10, stepsize=1e-3), **general_params),
            dict(logdir='./log/linearvf-kl2e-3-seed2', seed=2, desired_kl=2e-3, vf_type='linear', n_iter=500000, vf_params={}, **general_params),
            dict(logdir='./log/nnvf-kl2e-3-seed2', seed=2, desired_kl=2e-3, vf_type='nn', n_iter=300000, vf_params=dict(n_epochs=10, stepsize=1e-3), **general_params),
        ]
        import multiprocessing
        p = multiprocessing.Pool()
        p.map(main_pendulum1, params)

if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('module',choices=['0','1','2'],
                        help='0: cart-pole with linear v-func; '
                            +'1: cart-pole with neural-net v-func; '+
                             '2: pendulum')
    args=parser.parse_args()
    run(int(args.module))
