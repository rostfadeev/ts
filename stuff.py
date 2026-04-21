
import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf





def compile_and_fit(model, window, patience=5, MAX_EPOCHS=20):
  # EarlyStopping - Stop training when a monitored metric has stopped improving.
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience, mode='min')
  
  # configure
  model.compile(loss = tf.keras.losses.MeanSquaredError(), optimizer = tf.keras.optimizers.Adam(), metrics = [ tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.RootMeanSquaredError() ])
  
  print ("window.train: " + str(window.train))
  
  # training
  history = model.fit(window.train, epochs = MAX_EPOCHS, validation_data = window.val, callbacks = [early_stopping])
  return history


