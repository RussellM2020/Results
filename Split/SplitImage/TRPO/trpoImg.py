#import sys
#sys.path.append('/home/russellm/Research/Results/Split')
#sandbox.rocky.tf.policies

from fileHandling import store
from fileHandling import clear

from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from point_image_env import PointImageEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.policies.gaussian_conv_policy import GaussianConvPolicy
from sandbox.rocky.tf.spaces.box import Box
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc.instrument import run_experiment_lite


def run_task(*_):
    env = TfEnv(normalize(PointImageEnv()))
    ac_dim = env.action_space.shape[0]
    #print(type(env.action_space))
    #print(Box)
    policy = GaussianConvPolicy(
        name = "5Conv",
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.

        conv_filters = [10, 5],


        conv_filter_sizes = [5, 3],
        conv_strides = [1,1],
        conv_pads = ["SAME", "SAME"],


        hidden_sizes=[ 32, 32, ac_dim],
        learn_std = False,
    )

    """
            A network is composed of several convolution layers followed by some fc layers, by calling ConvNetwork (defined in sandbox.rocky.tf.core.network)
            
            conv_filters: a list of numbers of convolution kernel
            conv_filter_sizes: a list of sizes (int) of the convolution kernels
            conv_strides: a list of strides (int) of the conv kernels
            conv_pads: a list of pad formats (either 'SAME' or 'VALID')
            hidden_nonlinearity: a nonlinearity from tf.nn, shared by all conv and fc layers
            hidden_sizes: a list of numbers of hidden units for all fc layers

            How layers are created
             for idx, conv_filter, filter_size, stride, pad in zip(range(len(conv_filters)),conv_filters, conv_filter_sizes,conv_strides, conv_pads):
                create conv2D layer
                
    """
    baseline = ZeroBaseline(env_spec = env.spec)



    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=10000,
        max_path_length=100,
        n_itr=400,
        discount=0.99,
        plot=True
    )
    algo.train()


run_experiment_lite(

    run_task,
    # Number of parallel workers for sampling
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=3,
    #exp_name = "store1"
    #use_cloudpickle = False
    # plot=True,
    #logdir = "store1"
)