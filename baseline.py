
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


class Baseline(tf.keras.Model):
  def __init__(self, label_index=None):
    super().__init__()
    self.label_index = label_index

  def call(self, inputs):
    if self.label_index is None:
      return inputs
    
    result = inputs[:, :, self.label_index]
    return result[:, :, tf.newaxis]


class MultiStepLastBaseline(tf.keras.Model):
  def __init__(self, label_index=None, OUT_STEPS = 1):
    super().__init__()
    self.label_index = label_index
    self.OUT_STEPS = OUT_STEPS
  
  def call(self, inputs):
    return tf.tile(inputs[:, -1:, :], [1, self.OUT_STEPS, 1])


class RepeatBaseline(tf.keras.Model):
  def __init__(self, label_index=None, OUT_STEPS = 1):
    super().__init__()
    self.label_index = label_index
    self.OUT_STEPS = OUT_STEPS
  
  def call(self, inputs):
    return inputs


