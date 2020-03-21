import gym
import numpy as np
import yaml
import os
import random
from collections import OrderedDict
import tensorflow as tf
from stable_baselines.a2c.utils import conv, linear, conv_to_fc
from stable_baselines.common.tf_util import conv2d
from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import register_policy
from stable_baselines import PPO1


# # Custom Policy for CNN_small 
def cnn_small(X, **conv_kwargs):
    h = tf.cast(X, tf.float32) / 255.
    activ = tf.nn.relu
    h = activ(conv(h, 'c1', n_filters=8, filter_size=8, stride=4, init_scale=np.sqrt(2), **conv_kwargs))
    h = activ(conv(h, 'c2', n_filters=16, filter_size=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    h = conv_to_fc(h)
    network_fn = activ(linear(h, 'fc1', n_hidden=128, init_scale=np.sqrt(2)))
    
    return network_fn


class CustomPolicyCNNsmall(FeedForwardPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(CustomPolicyCNNsmall, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                cnn_extractor=cnn_small, feature_extraction="cnn", **_kwargs)


register_policy('CustomPolicyCNNsmall', CustomPolicyCNNsmall)



# # Custom Policy for CNN_paper
def cnn_paper(X, **conv_kwargs):
    h = tf.cast(X, tf.float32) / 255.
    activ = tf.nn.relu
    h = activ(conv(h, 'c1', n_filters=16, filter_size=8, stride=4, init_scale=np.sqrt(2), **conv_kwargs))
    h = activ(conv(h, 'c2', n_filters=32, filter_size=3, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    h = conv_to_fc(h)
    network_fn = activ(linear(h, 'fc1', n_hidden=128, init_scale=np.sqrt(2)))
    
    return network_fn


class CustomPolicyCNNpaper(FeedForwardPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(CustomPolicyCNNpaper, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                cnn_extractor=cnn_paper, feature_extraction="cnn", **_kwargs)

register_policy('CustomPolicyCNNpaper', CustomPolicyCNNpaper)




# # Custom Policy for CNN_Nvidia
def cnn_nvidia(X, **conv_kwargs):
    h = tf.cast(X, tf.float32) / 255.
    activ = tf.nn.relu
    h = activ(conv(h, 'c1', n_filters=24, filter_size=5, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    h = activ(conv(h, 'c2', n_filters=36, filter_size=5, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    h = activ(conv(h, 'c3', n_filters=48, filter_size=5, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    h = activ(conv(h, 'c4', n_filters=64, filter_size=3, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    h = activ(conv(h, 'c5', n_filters=64, filter_size=3, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
    h = conv_to_fc(h)
    h = activ(linear(h, 'fc1', n_hidden=1164, init_scale=np.sqrt(2)))
    network_fn = activ(linear(h, 'fc2', n_hidden=128, init_scale=np.sqrt(2)))
    
    return network_fn


class CustomPolicyCNNNvidia(FeedForwardPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(CustomPolicyCNNNvidia, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                cnn_extractor=cnn_nvidia, feature_extraction="cnn", **_kwargs)



register_policy('CustomPolicyCNNNvidia', CustomPolicyCNNNvidia)
