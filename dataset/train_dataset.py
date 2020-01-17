########################################




########################################


import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from utils.config import Config

class train_dataset():
    """
    Dataset class for train and validation split
    """
    def __init__(self,config):
        # initialize all locations
        training_csv_file = os.path.join(config.train_location,"training.csv")
        pass


    def copy_over(self):
        # function that takes care of moving dataset to localscratch
        pass

    def minibatch_gen(self):
        # generator that returns a minibatch
        pass
    