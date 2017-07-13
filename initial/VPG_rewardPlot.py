

#For cartpole
#discrete space (2,)
from rllab.envs.box2d.cartpole_env import CartpoleEnv
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
		        
        return logProb, Prob





K= 100000
#K is total number of experiments
#N = 10
#N is number of trajectories in one experiment
T = 100
#T is number of timesteps in one trajectory




env = CartPoleEnv()
inputShape = env.observation_space.shape

def sample(vector):
    #for discrete action space with 2 actions
    a = np.random.rand()
    if a<=vector[0]:
        return 0
    return 1





def  sampleTrajectory():

    observation_History=np.empty(shape = (1,1)+inputShape, dtype = float)
    action_History = np.empty(shape=1, dtype = float)
    reward_History=np.zeros(shape=1, dtype = float)
    


    observation = env.reset()
    #The observation is reset for the new trajectory

    
    #The sum of rewards needs to be reset for each separate trajectory.
    for timestep in range(T):


        
        observation_History = np.vstack((observation_History,[[observation]]))
 
        movement= session.run(PolicyLogProb, feed_dict = {x_ph: [[observation]]})

        action = sample(movement[0])
        #print(str(movement)+"   "+str(action))
       
        next_observation, new_reward, _,_ = env.step(action)
            
            
            
            # Placeholders take inputs of shape (?,1,4)
            # While collection, each observation is of shape (4,)
           
        #actionProb_History[timestep] = movement[action]

        action_History = np.concatenate((action_History,[action]))
        

        
        reward_History = np.concatenate((reward_History,[new_reward]))

            #new_rewardSum = np.sum(rewardSum_History)+(new_reward*discount)
            #rewardSum_History= np.concatenate((rewardSum_History, [new_rewardSum]))

        observation = next_observation

        if (env.steps_beyond_done !=None):
            #This is the condition where the pole has fallen. So we terminate the trajectory
            return endSample(observation_History, reward_History, action_History)

    
    return endSample(observation_History, reward_History, action_History)

def endSample(observation_History, reward_History, action_History):

    observation_History = np.delete(observation_History,(0),axis = 0)
    action_History = np.delete(action_History,(0))
    reward_History = np.delete(reward_History,(0))

    return observation_History, reward_History, action_History


x_ph = tf.placeholder(tf.float32, shape = ([None]+[1]+list(inputShape)))
a_ph = tf.placeholder(tf.float32, shape =[None])
aprob_ph = tf.placeholder(tf.float32, shape = [None])
rsum_ph = tf.placeholder(tf.float32, shape=[None])


PolicyLogProb, PolicyProb = policy(x_ph)

act0logProb, act1logProb = tf.split(PolicyLogProb,2,1)

ones_ph = tf.placeholder(dtype = tf.float32, shape=[None])
logProb = tf.multiply(tf.subtract(ones_ph, a_ph), act0logProb) + tf.multiply(a_ph, act1logProb)


_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Policy')

optimizer = tf.train.AdamOptimizer()



objective = tf.reduce_mean( tf.multiply(logProb, rsum_ph) )
 

train_policy = optimizer.minimize(loss = -objective, var_list = _vars)

session.run(tf.global_variables_initializer())



def sum_discounted_Rewards(rewards):
    Tmax = np.size(rewards)
    Qvalues = np.zeros(shape = Tmax)
    for timestep in range(Tmax):
        sum = 0
        for i in range(timestep,Tmax):
            sum+=rewards[i]
        Qvalues[timestep] = sum

    return Qvalues

objective_History=[]
RewardSum_History=[]
for experiment in range(K):

    
    obsH, rewards, actH  = sampleTrajectory()
    Qvalues = sum_discounted_Rewards(rewards)
#    Vvalues = np.ones(shape = np.size(rewards))*50
 #   Avalues = Qvalues - Vvalues
    RewardSum_History.append(Qvalues[0])

    ones = np.ones(shape = np.size(rewards))
    
    
    session.run(train_policy, feed_dict={x_ph: obsH, a_ph:actH, rsum_ph :Qvalues, ones_ph: ones})

    
    calculated_objective = session.run(objective, feed_dict={x_ph : obsH, a_ph:actH, rsum_ph: Qvalues, ones_ph:ones})
    

    objective_History.append(calculated_objective)
    if (experiment%1000 ==0):
                
        print("At the end of experiment "+ str(experiment) + "Sum of Rewards is  "+str(Qvalues[0]))
        
        print("At the end of experiment "+ str(experiment) + "Objective is  "+str(calculated_objective))
       
        print(actH)




Iterations= range(0,100000)
plt.plot(Iterations,RewardSum_History, '-r')
plt.plot(Iterations, [0]*100000, '-b')


plt.xlabel("Iteration")
plt.ylabel("Sum of Rewards")
plt.axis([0, 100000,0, 100])
plt.title('Vanilla Policy Gradient')
plt.savefig('RewardPlot1.png')






        



