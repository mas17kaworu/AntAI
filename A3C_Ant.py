import multiprocessing as mp
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
ENTROPY_BETA = 0.001
LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic

N_S = env.observation_space.shape[0]		# unknow
N_A = env.action_space.n  # 5