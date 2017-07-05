#For cartpole
#discrete space (2,)
#from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.envs.gym_env import GymEnv
from gym.envs.classic_control.cartpole import CartPoleEnv
import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
session = tf.InteractiveSession()
def policy(obs):
    with tf.variable_scope("Policy"):
        out =obs
        
       
        out = layers.convolution2d(out, num_outputs=10, kernel_size=3, stride=1, padding = "SAME",activation_fn=tf.nn.relu, trainable= True,
                                    weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=True, seed=None, dtype=tf.float32))
        out = layers.flatten(out)
    
        out = layers.fully_connected(out, num_outputs=5, activation_fn=tf.nn.relu, trainable = True, weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        logits = layers.fully_connected(out, num_outputs=2, activation_fn=None, trainable = True, weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        logProb = tf.nn.log_softmax(logits)
        Prob = tf.nn.softmax(logits)
                
        return logProb, Prob, logits
experiments = 300
#K is total number of experiments
batchSize = 100
#N is number of trajectories in one experiment
maxTrajLength = 100
#T is number of timesteps in one trajectory
totalTrajectories = experiments*batchSize
env = CartPoleEnv()
inputShape = env.observation_space.shape
#vf = LinearValuefunc()

def sample(vector):
    #for discrete action space with 2 actions
    a = np.random.rand()
    if a<=vector[0]:
        return 0
    return 1

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

def categorical_sample_logits(logits):
    U = tf.random_uniform(tf.shape(logits))
    return tf.argmax(logits -tf.log(-tf.log(U)), dimension = 1)

class LinearValuefunc:
    coef = None
    def fit(self, X, y):
        Xp = np.reshape(np.asarray(X),(np.shape(X)[0],inputShape[0]))
         #Here X is a list [?,1,4] #Here X is a list [?,1,4nputShape)) 
        A = Xp.T.dot(Xp)
        b = Xp.T.dot(y)
        self.coef = np.linalg.solve(A,b)
    def predict(self, X):
        #X is one observation
        if self.coef is None:

            
            return 0
        else:
            result = X.dot(self.coef)
            
            return result
    
   

def  sampleBatch():



   # observation_History=np.empty(shape = (1,1)+inputShape, dtype = float)    
   # action_History = np.empty(shape=1, dtype = float)
   # Qvalues_History = np.zeros(shape=1, dtype=float)
    trajectoryRewardSum = []
    observation_History = []
    action_History = []
    Qvalues_History = []
    Vvalues_History=[]   
    def sampleTrajectory(observation_History, action_History, Vvalues_History):
        trajTimesteps = 0
        rewards=[]
        traj_obs=[]
        observation = env.reset()
        #The observation is reset for the new trajectory
        #      The sum of rewards needs to be reset for each separate trajectory.
        for timestep in range(maxTrajLength):
            
            trajTimesteps+=1
            observation_History.append([observation])
            
            Vvalues_History.append(vf.predict(observation))

            #observation_History = np.vstack((observation_History,[[observation]]))
 
            action= session.run(sampled_ac, feed_dict = {x_ph: [[observation]]})

            #print(str(movement)+"   "+str(action))
       
            next_observation, new_reward, _,_ = env.step(action)
            
            
            
            # Placeholders take inputs of shape (?,1,4)
            # While collection, each observation is of shape (4,)
           
            #actionProb_History[timestep] = movement[action]
            action_History.append(action)
            #action_History = np.concatenate((action_History,[action]))
        
        
            #reward_History = np.concatenate((reward_History,[new_reward]))
            rewards.append(new_reward)
            #new_rewardSum = np.sum(rewardSum_History)+(new_reward*discount)
            #rewardSum_History= np.concatenate((rewardSum_History, [new_rewardSum]))
            observation = next_observation
            if (env.steps_beyond_done !=None):
            #This is the condition where the pole has fallen. So we terminate the trajectory
                return  rewards, trajTimesteps
                #return endSample(observation_History, reward_History, action_History)
        return  rewards, trajTimesteps


    batchTimeSteps=0

    for trajectory in range(batchSize):
        rewards, trajTimesteps = sampleTrajectory(observation_History, action_History, Vvalues_History)

         
       
       
       
       
        
        trajectoryRewardSum.append(np.sum(rewards))
        batchTimeSteps+=trajTimesteps
        compute_Qvalues(rewards,Qvalues_History )

        



       # V_value = vf.predict(observations)
        
   
    
    

    Advantage_History = np.subtract(Qvalues_History, Vvalues_History)
  
   
    return observation_History, Advantage_History, Qvalues_History,  action_History, trajectoryRewardSum,batchTimeSteps

def compute_Qvalues(rewards, Qvalues_History ):
    Tmax = np.size(rewards)
    returns=[]
    for timestep in range(Tmax):
        accum_rew = 0
        for i in range(timestep,Tmax):
            accum_rew+=rewards[i]
        #Qvalues_History = np.concatenate((Qvalues_History,[accum_rew]))
        #returns.append(accum_rew)
        Qvalues_History.append(accum_rew)
    #return returns   
    
x_ph = tf.placeholder(tf.float32, shape = ([None]+[1]+list(inputShape)))
a_ph = tf.placeholder(tf.float32, shape =[None])



aprob_ph = tf.placeholder(tf.float32, shape = [None])
rsum_ph = tf.placeholder(tf.float32, shape=[None])
PolicyLogProb, PolicyProb, PolicyLogits = policy(x_ph)

sampled_ac = categorical_sample_logits(PolicyLogits)[0]

act0logProb, act1logProb = tf.split(PolicyLogProb,2,1)
ones_ph = tf.placeholder(dtype = tf.float32, shape=[None])
#logProb = tf.multiply(tf.subtract(ones_ph, a_ph), act0logProb) + tf.multiply(a_ph, act1logProb)

sy_n = tf.shape(x_ph)[0]
logProb = fancy_slice_2d(PolicyLogProb, tf.range(sy_n), a_ph)

_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Policy')
optimizer = tf.train.AdamOptimizer()
objective = tf.reduce_mean( rsum_ph * logProb)
 
train_policy = optimizer.minimize(loss = -objective, var_list = _vars)
session.run(tf.global_variables_initializer())
objective_History=[]
RewardSum_History=[]

vf = LinearValuefunc()
for experiment in range(experiments):

    
    
    obsH, Avalues,Qvalues, actH , rewardSum, batchTimeSteps = sampleBatch()
    vf.fit(obsH, Qvalues)
    print("Size "+ str(np.size(obsH)))
    print("rewardSum "+str(rewardSum))

    ones = np.ones(batchTimeSteps)
    
    
    session.run(train_policy, feed_dict={x_ph: obsH, a_ph:actH, rsum_ph :Avalues, ones_ph: ones})
    
    calculated_objective = session.run(objective, feed_dict={x_ph : obsH, a_ph:actH, rsum_ph: Avalues, ones_ph:ones})
    
    objective_History.append(calculated_objective)
    #if (experiment%1000 ==0):
                
    print("At the end of experiment "+ str(experiment) + "Mean Sum of Rewards of latest Batch is  "+str(np.mean(rewardSum)))
    RewardSum_History.append(np.mean(rewardSum))     
    print("At the end of experiment "+ str(experiment) + "Objective is  "+str(calculated_objective))
       
        
Iterations= range(0,experiments)
plt.plot(Iterations,RewardSum_History, '-r')
plt.plot(Iterations, [0]*experiments, 'b-')
plt.xlabel("Iteration")
plt.ylabel("Sum of Rewards")
plt.axis([0, experiments,0, 100])
plt.title('Vanilla Policy Gradient')
plt.savefig('RewardPlotBaseline.png')
