import numpy as np
import tensorflow as tf
import gym
import logz
import scipy.signal
from point_image_env import PointImageEnv
from rllab.envs.normalized_env import normalize
import matplotlib.pyplot as plt
import time
import pickle
import os

def normc_initializer(std=1.0):
    """
    Initialize array with normalized columns
    """
    def _initializer(shape, dtype=None, partition_info=None): #pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def define_vars(size_in, size_out, name, weight_init):
    w = tf.get_variable(name + "/w", [size_in, size_out], initializer=weight_init)
    b = tf.get_variable(name + "/b", [size_out], initializer=tf.zeros_initializer)    
    return w,b

def define_conv_vars(filter_size, num_channels, num_filters, name, weight_init=tf.contrib.layers.xavier_initializer()):
    w = tf.get_variable(name + "/w", [filter_size, filter_size, num_channels, num_filters], initializer=weight_init)
    b = tf.get_variable(name + "/b", [num_filters], initializer=tf.zeros_initializer)    
    return w,b

def conv2d(img, w, b, strides=[1, 1, 1, 1]):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=strides, padding='SAME'), b))

def dense(x, w, b):
    """
    Dense (fully connected) layer
    """
    return tf.matmul(x, w) + b

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
    def __init__(self, ob_dim=1, **vf_params):#num_epochs=10, stepsize=1e-3):
        self.input_ph = tf.placeholder(shape=[None, ob_dim], name="ob_nn", dtype=tf.float32) # batch of observations
        self.value_ph = tf.placeholder(shape=[None,], name="val_nn", dtype=tf.float32)
        l1 = lrelu(dense(self.input_ph, 32, "h1_vf", weight_init=normc_initializer(1.0))) # hidden layer
        l2 = lrelu(dense(l1, 32, "h2_vf", weight_init=normc_initializer(1.0))) # hidden layer
        self.value = dense(l2, 1, "final_vf", weight_init=normc_initializer(0.1)) # "logits", describing probability distribution of final layer
        self.loss_vf = tf.reduce_sum(tf.nn.l2_loss(self.value_ph - self.value))
        self.minimizer = tf.train.AdamOptimizer(vf_params['stepsize']).minimize(self.loss_vf)
        self.num_epochs = vf_params['n_epochs']

    def fit(self, X, y):
        sess = tf.get_default_session()
        batchsize = 10
        num_data = len(X)
        num_batches = int(num_data/batchsize)
        idx = np.array(range(num_data))
        for i in range(self.num_epochs):
            #Do we need batchsize?
            np.random.shuffle(idx)
            for j in range(num_batches):
                bx = X[idx[j*batchsize: (j+1)*batchsize]]
                by = y[idx[j*batchsize: (j+1)*batchsize]]
                _, v_loss = sess.run([self.minimizer, self.loss_vf], feed_dict={self.input_ph: bx, self.value_ph: by})

    def predict(self, X):
        sess = tf.get_default_session()
        return sess.run(self.value, feed_dict={self.input_ph: X})[:,0]


class ZeroValueFunction(object):
    def fit(self, X, y):
        pass

    def predict(self, X):
        num_dps = len(X)
        return np.zeros((num_dps,))

def lrelu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)

def save_weights(sess, filename):
    vars = tf.trainable_variables()
    full_data = {}
    for v in vars:
        full_data[v.name] = sess.run(v)
    with open(filename,'wb') as f:
        pickle.dump(full_data, f)

def load_weights(sess, filename):
    vars = tf.trainable_variables()
    full_data = pickle.load( open(filename, "rb" ))
    for v in vars:
        assign_op = v.assign(full_data[v.name])
        sess.run(assign_op)
        print(v.name)

#Same structure baseline
def VPGImage_5(logdir, seed, n_iter, gamma, min_timesteps_per_batch, initial_stepsize, desired_kl, vf_params, vf_type="zero", animate=False, use_adaptive_stepsize=False, split_level=0, sup_learning_steps = 10):
    tf.set_random_seed(seed)
    np.random.seed(seed)

    env = normalize(PointImageEnv())

    ob_dim = env.observation_space.shape
    ac_dim = env.action_space.shape[0]
    logz.configure_output_dir(logdir)

    if vf_type == 'zero':
        vf = ZeroValueFunction()
    if vf_type == 'linear':
        vf = LinearValueFunction(**vf_params)
    elif vf_type == 'nn':
        vf = NnValueFunction(ob_dim=ob_dim, **vf_params)


    # Symbolic variables have the prefix sy_, to distinguish them from the numerical values
    # that are computed later in these function
    sy_ob_no = tf.placeholder(shape=[None] + list(ob_dim), name="ob", dtype=tf.float32) # batch of observations
    sy_ac_n = tf.placeholder(shape=[None, ac_dim], name="ac", dtype=tf.float32) # batch of actions taken by the policy, used for policy gradient computation
    sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32) # advantage function estimate
    sy_oldmean = tf.placeholder(shape=[None, ac_dim], name='oldmean', dtype=tf.float32) # logits BEFORE update (just used for KL diagnostic)
    logstd = tf.get_variable("logstdev", [ac_dim], initializer=tf.zeros_initializer)
    sy_oldlogstd = tf.placeholder(shape=[ac_dim], name='oldlogstd', dtype=tf.float32)
    sy_n = tf.shape(sy_ob_no)[0]



    param_lists = []
    num_filters = [10, 10]
    filter_size = [5, 5]
    

    #---------------------Conv Block 1-----------------------#

    w1, b1 = define_vars(32, 32, "h1", weight_init=normc_initializer(1.0))
    sy_h1 = lrelu(dense(critical_layer, w1, b1)) # hidden layer

    w2, b2 = define_vars(32, 32, "h2", weight_init=normc_initializer(1.0))
    sy_h2 = lrelu(dense(critical_layer, w2, b2)) # hidden layer

    w3, b3 = define_vars(32, 32, "h3", weight_init=normc_initializer(1.0))
    sy_h3 = lrelu(dense(critical_layer, w3, b3)) # hidden layer

     #---------------------Last 2 FC layers--------------------------#   
   
    w4, b4 = define_vars(32, 32, "h4", weight_init=normc_initializer(1.0))
    sy_h4 = lrelu(dense(critical_layer, w4, b4)) # hidden layer

    #w22, b22 = define_vars(32, 32, "h22", weight_init=normc_initializer(1.0))
    #sy_h22 = lrelu(dense(sy_h3, w4, b4)) # hidden layer

    wmean, bmean = define_vars(32, ac_dim, "mean", weight_init=normc_initializer(0.1))
    sy_mean = dense(sy_h4, wmean, bmean) # hidden layer

    
    #------------------------------------------------------------------#


    # act_var1 = tf.Variable(initial_value = tf.zeros([ min_timesteps_per_batch,32], dtype = tf.float32),name = "act_var1", trainable = True)
    # act_Op1 = act_var1.assign(critical_layer)

    # h21_back = lrelu(dense(act_var1, w21, b21)) 
        
    # sy_mean_back = dense(h21_back, wmean, bmean)
      
    



    
    sy_sampled_ac = sy_mean + tf.exp(logstd)*tf.random_normal(tf.shape(sy_mean), dtype=tf.float32) # sampled actions, used for defining the policy (NOT computing the policy gradient)
    

    #Computing log prob of the taken actions
    sy_logprob_n = -tf.reduce_sum(logstd) - ac_dim*0.5*tf.log(2*np.pi) - 0.5*tf.reduce_sum(tf.square(((sy_mean - sy_ac_n)/tf.exp(logstd))), reduction_indices=-1)
   

    sy_surr = - tf.reduce_mean(sy_adv_n * sy_logprob_n) # Loss function that we'll differentiate to get the policy gradient ("surr" is for "surrogate loss")
    #l2_loss = tf.nn.l2_loss(tf.subtract(act_var1, critical_layer))
        #activationOp will change during the PG_step
    #_SLvars = [w11, b11, w12, b12, w13, b13]
    _PGvars = [ w1, b1, w2, b2, w3, b3, w4, b4, wmean, bmean]
        #sy_stepsize = tf.placeholder(shape=[], dtype=tf.float32) # Symbolic, in case you want to change the stepsize during optimization. (We're not doing that currently)
        #tf.global_variables_initializer().run() #pylint: disable=E1101
          
    
    optimizerPG = tf.train.AdamOptimizer(1e-2)
    PG_step = optimizerPG.minimize(loss = sy_surr, var_list = _PGvars)
       
    # optimizerSL = tf.train.AdamOptimizer(1e-3)
    # SL_step = optimizerSL.minimize(loss =l2_loss, var_list = _SLvars)
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1) 
   

   

   
  
    # sy_kl = (tf.reduce_sum(logstd) - tf.reduce_sum(sy_oldlogstd)) +  0.5*tf.reduce_sum(tf.exp(2*(sy_oldlogstd - logstd))) - 0.5*ac_dim + \
    #         0.5*tf.reduce_sum(tf.square(((sy_oldmean - sy_mean))/tf.exp(logstd)))/tf.to_float(sy_n) 
    # # sy_kl_av1 = (tf.reduce_sum(logstd) - tf.reduce_sum(sy_oldlogstd)) +  0.5*tf.reduce_sum(tf.exp(2*(sy_oldlogstd - logstd))) - 0.5*ac_dim + \
    # #         0.5*tf.reduce_sum(tf.square(((sy_oldmean - sy_split_mean_av1))/tf.exp(logstd)))/tf.to_float(sy_n) 
    # #Computing entropy of the new policy
    # sy_ent = 0.5*ac_dim*tf.log(2*np.pi*np.e) + tf.reduce_sum(logstd) #Always the same
    # <<<<<<<<<<<<<print("Activation before PG")
       
    sess = tf.Session()
    sess.__enter__() 
    tf.global_variables_initializer().run() #initializing all the variables

    total_timesteps = 0
    stepsize = initial_stepsize #Need to use adaptive stepsize sometimes
    max_timesteps = 100 #TODO: Fix this to be general
    plt.ion()
    for i in range(n_iter):
        print("********** Iteration %i ************"%i)
        # Collect paths until we have enough timesteps
        timesteps_this_batch = 0
        flag = True
        paths = []
        while True:
            ob = env.reset()
            terminated = False
            obs, acs, rewards = [], [], []
            animate_this_episode=(len(paths)==0 and (i % 10 == 0) and animate)
            t = 0
            while True:
                if animate_this_episode:
                    env.render()
                obs.append(ob)
                ac = sess.run(sy_sampled_ac, feed_dict={sy_ob_no : ob[None]})[0] #TODO: Change this accordingly
                acs.append(ac)
                ob, rew, done, _ = env.step(ac)
                if animate_this_episode:
                    plt.imshow(ob)
                    plt.pause(0.0001)
                rewards.append(rew)
                t+= 1
                if done or t >= max_timesteps:
                    break 
            path = {"observation" : np.array(obs), "terminated" : terminated,
                    "reward" : np.array(rewards), "action" : np.array(acs)}
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch >= min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch
        # Estimate advantage function
        vtargs, vpreds, advs = [], [], []
        for path in paths:
            rew_t = path["reward"]
            return_t = discount(rew_t, gamma)
            # vpred_t = vf.predict(path["observation"])
            adv_t = return_t #- vpred_t
            advs.append(adv_t)
            vtargs.append(return_t)
            # vpreds.append(vpred_t)

        # Build arrays for policy update
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_n = np.concatenate([path["action"] for path in paths])
        adv_n = np.concatenate(advs)
        standardized_adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + 1e-8)
        vtarg_n = np.concatenate(vtargs)
        # vpred_n = np.concatenate(vpreds)
        # vf.fit(ob_no, vtarg_n)
        # Policy update
        # value = sess.run(act_Op1, feed_dict = {sy_ob_no:ob_no})
        # print("Activation before PG")
        # print(sess.run(act_var1))

        sess.run(PG_step, feed_dict={sy_ob_no:ob_no, sy_ac_n:ac_n, sy_adv_n:standardized_adv_n} )
        # print("Activation after PG")
        # print(sess.run(act_var1))
        # sess.run(SL_step, feed_dict={sy_ob_no:ob_no} )


        # sl_loss = 0
        # if split_level == 0:
        #     _, oldmean, oldlogstd = sess.run([update_op, sy_mean, logstd], feed_dict={sy_ob_no:ob_no, sy_ac_n:ac_n, sy_adv_n:standardized_adv_n, sy_stepsize:stepsize})
        #     #TODO: Need to do the split training here
        #     kl, ent = sess.run([sy_kl, sy_ent], feed_dict={sy_ob_no:ob_no, sy_oldmean:oldmean, sy_oldlogstd: oldlogstd})
        # elif split_level == 1:
        #     #assign via forward pass
        #     sess.run(assign_op, feed_dict={sy_ob_no:ob_no})
        #     _, oldmean, oldlogstd = sess.run([update_op_av1, sy_split_mean_av1, logstd], feed_dict={sy_ob_no:ob_no, sy_ac_n:ac_n, sy_adv_n:standardized_adv_n, sy_stepsize:stepsize})
        #     for _ in range(sup_learning_steps):
        #         _, sl_loss_curr = sess.run([match_opt_av1, loss_match_av1], feed_dict={sy_ob_no:ob_no, sy_stepsize:stepsize})
        #         sl_loss += sl_loss_curr
        #     sl_loss = sl_loss/10.0
        #     kl, ent = sess.run([sy_kl_av1, sy_ent], feed_dict={sy_ob_no:ob_no, sy_oldmean:oldmean, sy_oldlogstd: oldlogstd})


        # kl = kl[0]
        if use_adaptive_stepsize:
            if kl > desired_kl * 2: 
                stepsize /= 1.5
                print('stepsize -> %s'%stepsize)
            elif kl < desired_kl / 2: 
                stepsize *= 1.5
                print('stepsize -> %s'%stepsize)
            else:
                print('stepsize OK')

        # Log diagnostics
        logz.log_tabular("EpRewMean", np.mean([path["reward"].sum() for path in paths]))
        logz.log_tabular("EpLenMean", np.mean([pathlength(path) for path in paths]))
        #logz.log_tabular("KLOldNew", kl)
       # logz.log_tabular("Entropy", ent)
        #logz.log_tabular("SL Loss", sl_loss)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        logz.dump_tabular()
        save_weights(sess, "curr_weights.pkl")

if __name__ == "__main__":
    general_params = dict(gamma=0.97, animate=False, min_timesteps_per_batch=4000, n_iter=2000, initial_stepsize=1e-3)
    SplitImage(logdir=None, seed=0, desired_kl=2e-3, vf_type='linear', vf_params={}, use_adaptive_stepsize=False, split_level=0, sup_learning_steps = 1, **general_params)



