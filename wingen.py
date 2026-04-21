
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

from itertools import chain, islice
from more_itertools import windowed



class WindowGenerator():
  def __init__(self, input_width, label_width, shift, train_df, val_df, test_df, clims = None, train_dt = None, val_dt = None, test_dt = None, train_mean = None, train_std = None, label_columns = None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df   = val_df
    self.test_df  = test_df
    
    # Store day of year of the first index
    self.train_dt = train_dt
    self.val_dt   = val_dt
    self.test_dt  = test_dt
    
    self.clims = clims
    
    # Work out the label column indices.
    self.label_columns = label_columns
    
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
    
    self.column_indices = {name: i for i, name in enumerate(train_df.columns)}
    
    self.train_mean = None
    self.train_std  = None
    
    #if label_columns is not None:
    if train_mean is not None:
      self.train_mean = train_mean
    if train_std is not None:
      self.train_std = train_std
    
    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift       = shift
    
    self.total_window_size = input_width + shift
    
    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]
    
    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
    
    self.color = [ 'black', 'aqua', 'magenta', 'lime', 'gold' ]

  def __repr__(self):
    return '\n'.join([
        f'\nlabel_columns_indices: {self.label_columns_indices}',
        f'column_indices:        {self.column_indices}',
        f'Total window size:     {self.total_window_size}',
        f'Input indices:         {self.input_indices}',
        f'Label indices:         {self.label_indices}',
        f'Label column name(s):  {self.label_columns}\n'])
  
  def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    
    if self.label_columns is not None:
      labels = tf.stack(
          [labels[:, :, self.column_indices[name]] for name in self.label_columns],
          axis=-1)
    
    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])
    
    return inputs, labels
  
  def plot(self, models=None, plot_col = [ 'sst' ], max_subplots=6, title='', draw_bias = False):
    inputs, labels = self.example
    
    plt.figure(figsize=(12, 8))
    plot_col_index = [ self.column_indices[pc] for pc in plot_col ]
    max_n = min(max_subplots, len(inputs))
    
    if self.label_columns is None:
      label_col = plot_col
      label_col_index = self.label_columns_indices.get(plot_col[0], None)
    else:
      label_col = self.label_columns
      label_col_index = self.label_columns_indices.get(label_col[0], None)
    
    #if hasattr(label_col, "__len__"):
    #  label_col = label_col[0]
    
    plot_color = [ "blue", "aqua", "teal", "deepskyblue", "aquamarine", "royalblue" ]
    
    indx = None
    
    lclim = False
    if 'sst_clim' in self.column_indices:
      lclim = True
    
    for n in range(max_n):
      plt.subplot(max_n, 1, n+1)
      #plt.ylabel(f'{plot_col} [normed]')
      
      if draw_bias == False:
        data2draw = [ inputs [n, :, pci] for pci in plot_col_index ]
        if lclim:
          clim2draw = inputs [n, :, self.column_indices['sst_clim']]
        
        if self.train_std is not None and self.train_mean is not None:
          for i in range(len(plot_col)):
            data2draw[i] = data2draw[i]*self.train_std[plot_col[i]] + self.train_mean[plot_col[i]]
          if lclim:
            lc = 'sst'
            clim2draw = clim2draw*self.train_std[lc] + self.train_mean[lc]
        
        for i in range(len(plot_col)):
          plt.plot(self.input_indices, data2draw[i], label = 'Inputs ' + str(plot_col[i]), linestyle='-', color = plot_color[i])
        if lclim:
          plt.plot(self.input_indices, clim2draw, label='Clim mean', linestyle='-', color = 'pink')
        
        #=================
        indx = None
        if True and lclim:
          ncl = self.input_width
          
          indx = 0
          if ncl >= len(self.clims):
            ncl = len(self.clims)-1
            indx -= len(clim2draw)+1
          
          result = list(windowed(chain(self.clims, islice(self.clims, ncl-1)), ncl))
          
          zlast = clim2draw[len(clim2draw)-ncl:len(clim2draw)]
          
          indx += np.argmin( [ np.sqrt(((np.array(r) - np.array(zlast))**2).sum(-1).mean()) for r in result ] )
          
          if False:
            print ("indx:        " + str(indx))
            print ("input_width: " + str(self.input_width))
            print ("ncl:         " + str(ncl))
            print ("label_width: " + str(self.label_width))
            print ("clim2draw:   " + str(len(clim2draw)))
          
          li = [i for i in range(indx,indx + self.input_width + self.label_width)]
          for j in range(len(li)):
            if li[j] < 0:
              li[j] += len(self.clims)
            if li[j] < 0:
              li[j] += len(self.clims)
            if li[j] >= len(self.clims):
              li[j] -= len(self.clims)
            if li[j] >= len(self.clims):
              li[j] -= len(self.clims)
          
          clim2draw = [ self.clims[i] for i in li ]
        #=================
      
      # draw labels
      
      print ("plot_col:        " + str(plot_col))
      print ("label_col:       " + str(label_col))
      print ("plot_col_index:  " + str(plot_col_index))
      print ("label_col_index: " + str(label_col_index))
      
      if label_col_index is None:
        continue
      
      if draw_bias == False:
        data2draw = labels[n, :, label_col_index]
        
        if self.train_std is not None and self.train_mean is not None:
          for i in range(len(label_col)):
            data2draw = data2draw*self.train_std[label_col[label_col_index]] + self.train_mean[label_col[label_col_index]]
          
        if len(data2draw) > 1:
          plt.plot(self.label_indices, data2draw, label='Labels ' + str(label_col[label_col_index]), linestyle='-', c = 'green')
        else:
          plt.plot(self.label_indices, data2draw, label='Labels ' + str(label_col[label_col_index]), marker='.', c = 'green')
      
      if models is not None:
        data2draw_p = []
        for m in range(len(models)):
          data2draw_p.append(models[m]["model"](inputs)[n, :, label_col_index])
        
        data2draw_l = labels[n, :, label_col_index]
        
        if self.train_std is not None and self.train_mean is not None:
          lc = label_col[0]
          for m in range(len(models)):
            data2draw_p[m] = data2draw_p[m]*self.train_std[lc] + self.train_mean[lc]
          data2draw_l = data2draw_l*self.train_std[lc] + self.train_mean[lc]
        
        for m in range(len(models)):
          if draw_bias == True:
            data2draw = data2draw_p[m] - data2draw_l
            plt_label = "Prediction bias"
            
            if True:
              plt.axhline(y = -1., color='grey', linestyle='--', lw = 1)
              plt.axhline(y =  0., color='grey', linestyle='--', lw = 1)
              plt.axhline(y =  1., color='grey', linestyle='--', lw = 1)
            
          else:
            data2draw = data2draw_p[m]
            #plt_label = "Predictions"
            plt_label = models[m]["title_short"]
          
          if len(data2draw) > 1:
            plt.plot(self.label_indices, data2draw,   label=plt_label,   linestyle='-', c = self.color[m])
          else:
            plt.plot(self.label_indices, data2draw,   label=plt_label,   marker='.', c = self.color[m])
        
        if indx is not None:
          plt.plot([ i for i in range(len(clim2draw)) ], clim2draw, label='Clim mean2', linestyle='-', c = 'grey')
      
      if n == 0:
        plt.legend()
      plt.title(title)

    plt.xlabel('Time [day]')
    
  def make_dataset(self, data):
    data = np.array(data, dtype=np.float64)
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=32,)
    
    ds = ds.map(self.split_window)
    
    return ds

  @property
  def train(self):
    return self.make_dataset(self.train_df)
  
  @property
  def val(self):
    return self.make_dataset(self.val_df)
  
  @property
  def test(self):
    return self.make_dataset(self.test_df)

  @property
  def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    
    if result is None:
      # No example batch was found, so get one from the `.train` dataset
      result = next(iter(self.train))
      # And cache it for next time
      self._example = result
    return result


