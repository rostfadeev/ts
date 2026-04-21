#!/home/rrd/rrdrio/util/miniconda/envs/ts03/bin/python

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

import xarray as xr

from itertools import chain, islice
from more_itertools import windowed

import wingen as wg
import baseline as bl
import stuff
import ar_lstm
import transformer


pd.set_option("display.precision", 16)
pd.set_option("styler.format.precision", 16)
tf.keras.backend.set_floatx('float64')

L_baseline = False
L_linear = False
L_dense = False
L_multi_step_dense = False
L_сonvolution_neural_network = False
L_recurrent_neural_network = False

L_mlst_baseline = False 
L_mlst_repeat_baseline = False
L_mlst_linear = False
L_mlst_dense = True
L_mlst_сonvolution_neural_network = True
L_mlst_recurrent_neural_network = True
L_mlst_2level_recurrent_neural_network =False
L_mlst_gated_recurrent_units = False

L_mlst_autoregressive_recurrent_neural_network = False

L_mlst_bidirectional_lstm = False
L_mlst_transformer = False

multi_window_models = []

draw_bias = False
single_plot = False

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


dirin = "./" #home/rrd/rrdrio/workflow/era5/cdsapi/"

fnin = "erfclim.10013000_00-h500.nc"

varid = [ "sst", "mslp", "tp" ]
#years = [ 1991, 1992 ]

pc_train = 0.5
pc_val   = 0.3
pc_test  = 0.2

MAX_EPOCHS = 20

INPUT_WIDTH = 800
OUT_STEPS   = 240

INPUT_WIDTH = 600
OUT_STEPS   = 120

if L_mlst_repeat_baseline:
  OUT_STEPS = INPUT_WIDTH

CONV_WIDTH  = 400

# for  L_сonvolution_neural_network
LABEL_WIDTH = INPUT_WIDTH - (CONV_WIDTH - 1) 
#INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)


max_subplots = 2


df = pd.DataFrame()


xr.open_dataset(fnin, engine="netcdf4")




exit()

for var in varid:
  if fnin.find("<var>") >= 0:
    fn = fnin.replace("<var>", var)
    
    print ("read file: " + str(dirin + "/" + fn))
    
    dfl = pd.read_csv(dirin + "/" + fn)
    dfl = dfl.rename(columns = { 'value' : var })
    
    dtl = dfl.iloc[:,0].values
    
    if df.empty:
      dt = dtl
      df = dfl
    else:
      if not (dtl==dt).all():
        print ("Error! dtl != dt")
        exit()
      
      df.insert(1, var, dfl[var])
    
    
print ("time records found: " + str(len(dt)))
print (df)

df[varid[0]] = df[varid[0]] - float(273.15)

date_time = pd.to_datetime(df.pop('date'), format='%Y-%m-%d')

clim_mean   = np.zeros((366), dtype = 'float')
meancounter = np.zeros((366), dtype = 'i')

for day in range(366):
  for i in range(len(date_time)):
    if date_time[i].day_of_year-1 == day:
      clim_mean[day] += df[varid[0]][i]
      meancounter[day] += 1

for day in range(366):
  clim_mean[day] /= float(meancounter[day])

#print (meancounter)
#print (clim_mean)

cm = np.zeros(len(df[varid[0]]))
for i in range(len(df)):
  cm[i] = clim_mean[date_time[i].day_of_year-1]

var = "sst_clim"
varid.append(var)

df.insert(len(varid)-1, var, cm)

print (varid)
print (df)

#exit()


if False:
  plt.figure(figsize=(12, 8))
  plt.title("clim mean of " + varid[0])
  plt.plot(clim_mean)
  
  plt.show()
  exit()


plot_cols = varid
plot_features = df[plot_cols]
plot_features.index = date_time
plot_features.plot(subplots=True)
#plt.show()



# filtering
if False:
  wv = df['sst']
  bad_wv = wv == -9999.0
  wv[bad_wv] = 0.0

# converting to seconds
timestamp_s = date_time.map(pd.Timestamp.timestamp)
#print (timestamp_s)


# split the data
column_indices = {name: i for i, name in enumerate(df.columns)}

if abs(pc_train + pc_val + pc_test -1.) > 0.000000001:
  print ("Error! Sum of pc_* != 1")
  print (str(pc_train + pc_val + pc_test))
  exit()

n = len(df)
train_df = df[0:int(n*pc_train)]
val_df   = df[int(n*pc_train):int(n*(1. - pc_test))]
test_df  = df[int(n*(1. - pc_test)):]

train_dt = date_time[0].day_of_year-1
val_dt   = date_time[int(n*pc_train)].day_of_year-1
test_dt  = date_time[int(n*(1. - pc_test))].day_of_year-1

print (train_dt)
print (val_dt)
print (test_dt)

num_features = df.shape[1]

print ("pc_train/pc_val/pc_test = " + str(pc_train) + "/" + str(pc_val) + "/" + str(pc_test))


# Normalize the data
train_mean = train_df.mean()
train_std = train_df.std()

print ("------------------===")
print (train_std)
print (train_mean)

train_df = (train_df - train_mean) / train_std
val_df   = (val_df   - train_mean) / train_std
test_df  = (test_df  - train_mean) / train_std

df_std = (df - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')

plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
#ax.set_xticklabels(df.keys(), rotation=90)
#qplt.show()


if False:
  w1 = wg.WindowGenerator(input_width=24, label_width=1, shift=24, train_df = train_df, val_df = val_df, test_df = test_df, label_columns=[varid[0]])
  print (w1)


  # check: stack three slices, the length of the total window.
  example_window = tf.stack([np.array(train_df[:w1.total_window_size]),
                            np.array(train_df[100:100+w1.total_window_size]),
                            np.array(train_df[200:200+w1.total_window_size])])

  #print ("example_window  \n" + str(train_df[:w1.total_window_size]))
  #print (example_window)

  example_inputs, example_labels = w1.split_window(example_window)

  print('All shapes are: (batch, time, features)')
  print(f'Window shape: {example_window.shape}')
  print(f'Inputs shape: {example_inputs.shape}')
  print(f'Labels shape: {example_labels.shape}')

  w1.example1 = example_inputs, example_labels

  #for var in varid:
  #  w1.plot(plot_col = var)

  #plt.show()


  single_step_window = wg.WindowGenerator(input_width=24, label_width=1, shift=24,
                                          train_df = train_df, val_df = val_df, test_df = test_df,
                                          train_mean = train_mean, train_std = train_std,
                                          label_columns=[ varid[0] ])

  print ("\n single_step_window \n" + str(single_step_window))


  single_step_window

  w1.train.element_spec

  print (single_step_window.train)


  for example_inputs, example_labels in single_step_window.train.take(1):
    print (single_step_window.train.take(1))
    print ("++")
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')



val_performance = {}
performance = {}

multi_val_performance = {}
multi_performance = {}

single_step_window = wg.WindowGenerator(input_width=1, label_width=1, shift=1,
                                        train_df = train_df, val_df = val_df, test_df = test_df,
                                        clims = clim_mean,
                                        train_mean = train_mean, train_std = train_std,
                                        label_columns = [ varid[0] ])

wide_window = wg.WindowGenerator(input_width=INPUT_WIDTH, label_width=INPUT_WIDTH, shift=1,
                                 train_df = train_df, val_df = val_df, test_df = test_df,
                                 clims = clim_mean,
                                 train_mean = train_mean, train_std = train_std,
                                 label_columns = [ varid[0] ])

conv_window = wg.WindowGenerator(input_width=CONV_WIDTH, label_width=1, shift=1,
                                 train_df = train_df, val_df = val_df, test_df = test_df,
                                 clims = clim_mean,
                                 train_mean = train_mean, train_std = train_std,
                                 label_columns = [ varid[0] ])
wide_conv_window = wg.WindowGenerator(
                                input_width=INPUT_WIDTH, label_width=LABEL_WIDTH, shift=1,
                                train_df = train_df, val_df = val_df, test_df = test_df,
                                clims = clim_mean,
                                train_dt = train_dt, val_dt = val_dt, test_dt = test_dt,
                                train_mean = train_mean, train_std = train_std,
                                label_columns = [ varid[0] ])

multi_window = wg.WindowGenerator(
                                input_width=INPUT_WIDTH, label_width=OUT_STEPS, shift=OUT_STEPS,
                                train_df = train_df, val_df = val_df, test_df = test_df,
                                clims = clim_mean,
                                train_dt = train_dt, val_dt = val_dt, test_dt = test_dt,
                                train_mean = train_mean, train_std = train_std,
                                label_columns = [ varid[0] ] )


############## baseline


if L_baseline:
  baseline = {
    "title_long":  "Baseline",
    "title_short": "baseline",
    "model": bl.Baseline(label_index=column_indices[ varid[0] ])
  }
  print(); print (baseline["title_long"]); print()
  
  baseline["model"].compile(loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanAbsoluteError()])

  val_performance['Baseline'] = baseline["model"].evaluate(single_step_window.val, return_dict=True)
  performance['Baseline']     = baseline["model"].evaluate(single_step_window.test, verbose=0, return_dict=True)
  
  wide_window

  print('Input shape:', wide_window.example[0].shape)
  print('Output shape:', baseline["model"](wide_window.example[0]).shape)

  wide_window.plot(models = [ baseline ], max_subplots = max_subplots, title = baseline["title_long"])


if L_mlst_baseline:
  last_baseline = {
    "title_long":  "Multi step baseline",
    "title_short": "mlst_bl",
    "model": bl.MultiStepLastBaseline(label_index=column_indices[ varid[0] ], OUT_STEPS = OUT_STEPS)
  }
  print(); print (last_baseline["title_long"]); print()
  
  last_baseline["model"].compile(loss=tf.keras.losses.MeanSquaredError(),
                        metrics=[tf.keras.metrics.MeanAbsoluteError()])

  multi_val_performance['Last'] = last_baseline["model"].evaluate(multi_window.val, return_dict=True)
  multi_performance['Last'] = last_baseline["model"].evaluate(multi_window.test, verbose=0, return_dict=True)
  
  multi_window_models.append( last_baseline )
  
  if single_plot:
    multi_window.plot(models = [ last_baseline ], max_subplots = max_subplots, title=last_baseline["title_long"])


if L_mlst_repeat_baseline:
  
  if INPUT_WIDTH != OUT_STEPS:
    print ("error: INPUT_WIDTH must equal to OUT_STEPS in case of L_mlst_repeat_baseline")
    exit()
  
  repeat_baseline = {
    "title_long":  "Multi step repeat baseline",
    "title_short": "mlst_rbl",
    "model": bl.RepeatBaseline()
  }
  print(); print (repeat_baseline["title_long"]); print()
  
  repeat_baseline["model"].compile(loss=tf.keras.losses.MeanSquaredError(),
                          metrics=[tf.keras.metrics.MeanAbsoluteError()])
  
  multi_val_performance['Repeat'] = repeat_baseline["model"].evaluate(multi_window.val, return_dict=True)
  multi_performance['Repeat'] = repeat_baseline["model"].evaluate(multi_window.test, verbose=0, return_dict=True)
  
  multi_window_models.append( repeat_baseline )
  
  if single_plot:
    multi_window.plot(models = [ repeat_baseline ], max_subplots = max_subplots, title=repeat_baseline["title_long"])


############## linear


if L_linear:
  linear = {
    "title_long":  "Linear",
    "title_short": "Linear",
    "model": tf.keras.Sequential([
      tf.keras.layers.Dense(units=1)
    ])
  }
  print(); print (linear["title_long"]); print()

  print('Input shape:', single_step_window.example[0].shape)
  print('Output shape:', linear["model"](single_step_window.example[0]).shape)

  history = stuff.compile_and_fit(linear["model"], single_step_window, MAX_EPOCHS=MAX_EPOCHS)
  
  linear["history"] = history
  
  val_performance['Linear'] = linear["model"].evaluate(single_step_window.val, return_dict=True)
  performance['Linear']     = linear["model"].evaluate(single_step_window.test, verbose=0, return_dict=True)

  wide_window.plot(models = [ linear ], max_subplots = max_subplots, title=linear["title_long"])

  if False:
    plt.figure(figsize=(12, 6))
    plt.bar(x = range(len(train_df.columns)), height=linear.layers[0].kernel[:,0].numpy())
    axis = plt.gca()
    axis.set_xticks(range(len(train_df.columns)))
    axis.set_xticklabels(train_df.columns, rotation=90)


if L_mlst_linear:
  multi_linear_model = {
    "title_long":  "Multi step linear",
    "title_short": "mlst_lin",
    "model": tf.keras.Sequential([
      # Take the last time-step.
      # Shape [batch, time, features] => [batch, 1, features]
      tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
      # Shape => [batch, 1, out_steps*features]
      tf.keras.layers.Dense(OUT_STEPS*num_features,
                            kernel_initializer=tf.initializers.zeros()),
      # Shape => [batch, out_steps, features]
      tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])
  }
  print(); print (multi_linear_model["title_long"]); print()
  
  history = stuff.compile_and_fit(multi_linear_model["model"], multi_window, MAX_EPOCHS=MAX_EPOCHS)
  
  multi_linear_model["history"] = history
  
  #IPython.display.clear_output()
  multi_val_performance['Linear'] = multi_linear_model["model"].evaluate(multi_window.val, return_dict=True)
  multi_performance['Linear'] = multi_linear_model["model"].evaluate(multi_window.test, verbose=0, return_dict=True)
  
  multi_window_models.append( multi_linear_model )
  
  if single_plot:
    multi_window.plot(models = [ multi_linear_model ], max_subplots = max_subplots, title = multi_linear_model["title_long"])



############## linear multy-layer, 1->1


if L_dense:
  dense = {
    "title_long":  "Dense",
    "title_short": "Dense",
    "model": tf.keras.Sequential([
      tf.keras.layers.Dense(units=64, activation='relu'),
      tf.keras.layers.Dense(units=64, activation='relu'),
      tf.keras.layers.Dense(units=1)
    ])
  }
  print(); print (dense["title_long"]); print()

  history = stuff.compile_and_fit(dense["model"], single_step_window, MAX_EPOCHS=MAX_EPOCHS)
  
  dense["history"] = history

  val_performance['Dense'] = dense["model"].evaluate(single_step_window.val, return_dict=True)
  performance['Dense']     = dense["model"].evaluate(single_step_window.test, verbose=0, return_dict=True)


  wide_window.plot(models = [ dense ], max_subplots = max_subplots, title=dense["title_long"])

  if False:
    plt.figure(figsize=(12, 6))
    plt.bar(x = range(len(train_df.columns)), height=dense.layers[0].kernel[:,0].numpy())
    axis = plt.gca()
    axis.set_xticks(range(len(train_df.columns)))
    axis.set_xticklabels(train_df.columns, rotation=90)

if L_mlst_dense:
  multi_dense_model = {
    "title_long":  "Multi step dense",
    "title_short": "mlst_dense",
    "model": tf.keras.Sequential([
      # Take the last time step.
      # Shape [batch, time, features] => [batch, 1, features]
      tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
      # Shape => [batch, 1, dense_units]
      tf.keras.layers.Dense(512, activation='relu'),
      # Shape => [batch, out_steps*features]
      tf.keras.layers.Dense(OUT_STEPS*num_features,
                            kernel_initializer=tf.initializers.zeros()),
      # Shape => [batch, out_steps, features]
      tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])
  }
  print(); print (multi_dense_model["title_long"]); print()

  history = stuff.compile_and_fit(multi_dense_model["model"], multi_window, MAX_EPOCHS=MAX_EPOCHS)
  
  multi_dense_model["history"] = history

  #IPython.display.clear_output()
  multi_val_performance['Dense'] = multi_dense_model["model"].evaluate(multi_window.val, return_dict=True)
  multi_performance['Dense'] = multi_dense_model["model"].evaluate(multi_window.test, verbose=0, return_dict=True)
  
  multi_window_models.append( multi_dense_model )
  
  if single_plot:
    multi_window.plot(models = [ multi_dense_model ], max_subplots = max_subplots, title = multi_dense_model["title_long"])



############## Multi-step dense, 3->1


if L_multi_step_dense:
  
  conv_window
  
  multi_step_dense = {
    "title_long":  "True multi step dense",
    "title_short": "true_mlst_dense",
    "model": tf.keras.Sequential([
      # Shape: (time, features) => (time*features)
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(units=32, activation='relu'),
      tf.keras.layers.Dense(units=32, activation='relu'),
      tf.keras.layers.Dense(units=32, activation='relu'),
      tf.keras.layers.Dense(units=32, activation='relu'),
      tf.keras.layers.Dense(units=32, activation='relu'),
      tf.keras.layers.Dense(units=1),
      # Add back the time dimension.
      # Shape: (outputs) => (1, outputs)
      tf.keras.layers.Reshape([1, -1]),
    ])
  }
  print(); print (multi_step_dense["title_long"]); print()

  print('Input shape:', conv_window.example[0].shape)
  print('Output shape:', multi_step_dense["model"](conv_window.example[0]).shape)

  history = stuff.compile_and_fit(multi_step_dense["model"], conv_window, MAX_EPOCHS=MAX_EPOCHS)
  
  multi_step_dense["history"] = history

  #IPython.display.clear_output()
  val_performance['Multi step dense'] = multi_step_dense["model"].evaluate(conv_window.val, return_dict=True)
  performance['Multi step dense'] = multi_step_dense["model"].evaluate(conv_window.test, verbose=0, return_dict=True)
  
  
  plt.suptitle("Given " + str(CONV_WIDTH) + " hours of inputs, predict 1 hour into the future.")
  conv_window.plot(models = [ multi_step_dense ], max_subplots = max_subplots, title=multi_step_dense["title_long"])

############## Convolution neural network


if L_сonvolution_neural_network:

  conv_window
  
  conv_model = {
    "title_long":  "Convolution neural network",
    "title_short": "conv",
    "model": tf.keras.Sequential([
      tf.keras.layers.Conv1D(filters=32,
                            kernel_size=(CONV_WIDTH,),
                            activation='relu'),
      tf.keras.layers.Dense(units=32, activation='relu'),
      tf.keras.layers.Dense(units=1),
    ])
  }
  print(); print (conv_model["title_long"]); print()

  print("Conv model on `conv_window`")
  print('Input shape:', conv_window.example[0].shape)
  print('Output shape:', conv_model["model"](conv_window.example[0]).shape)
  
  history = stuff.compile_and_fit(conv_model["model"], conv_window, MAX_EPOCHS=MAX_EPOCHS)
  
  conv_model["history"] = history
  
  #IPython.display.clear_output()
  val_performance['Conv'] = conv_model["model"].evaluate(conv_window.val, return_dict=True)
  performance['Conv'] = conv_model["model"].evaluate(conv_window.test, verbose=0, return_dict=True)
  
  wide_conv_window
  
  print("Wide conv window")
  print('Input shape:', wide_conv_window.example[0].shape)
  print('Labels shape:', wide_conv_window.example[1].shape)
  print('Output shape:', conv_model["model"](wide_conv_window.example[0]).shape)
  
  wide_conv_window.plot(models = [ conv_model ], max_subplots = max_subplots, title=conv_model["title_long"])
  
  if draw_bias:
    wide_conv_window.plot(models = [ conv_model ], max_subplots = max_subplots, draw_bias = draw_bias, title=conv_model["title_long"])




if L_mlst_сonvolution_neural_network:
  if CONV_WIDTH >= INPUT_WIDTH:
    print ("error: CONV_WIDTH must be decreased!")
    exit()
  
  multi_conv_model = { 
    "title_long":  "Multi step convolution neural network",
    "title_short": "mlst_cnn",
    "model": tf.keras.Sequential([
      # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
      tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
      # Shape => [batch, 1, conv_units]
      tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
      # Shape => [batch, 1,  out_steps*features]
      tf.keras.layers.Dense(OUT_STEPS*num_features,
                            kernel_initializer=tf.initializers.zeros()),
      # Shape => [batch, out_steps, features]
      tf.keras.layers.Reshape([OUT_STEPS, num_features])
  ])}
  print(); print (multi_conv_model["title_long"]); print()
  
  history = stuff.compile_and_fit(multi_conv_model["model"], multi_window, MAX_EPOCHS=MAX_EPOCHS)
  
  multi_conv_model["history"] = history
  
  multi_val_performance['Conv'] = multi_conv_model["model"].evaluate(multi_window.val, return_dict=True)
  multi_performance['Conv'] = multi_conv_model["model"].evaluate(multi_window.test, verbose=0, return_dict=True)
  
  multi_window_models.append( multi_conv_model )
  
  if single_plot:
    multi_window.plot(models = [ multi_conv_model ], max_subplots = max_subplots, title = multi_conv_model["title_long"])




############## Recurrent neural network


if L_recurrent_neural_network:
  lstm_model = {
    "title_long":  "Recurrent neural network (lstm)",
    "title_short": "lstm",
    "model": tf.keras.models.Sequential([
      # Shape [batch, time, features] => [batch, time, lstm_units]
      tf.keras.layers.LSTM(32, return_sequences=True),
      # Shape => [batch, time, features]
      tf.keras.layers.Dense(units=1)
    ])
  }
  print(); print (lstm_model["title_long"]); print()
  
  wide_window
  
  print('Input shape:', wide_window.example[0].shape)
  print('Output shape:', lstm_model["model"](wide_window.example[0]).shape)

  history = stuff.compile_and_fit(lstm_model["model"], wide_window, MAX_EPOCHS=MAX_EPOCHS)
  
  lstm_model["history"] = history
  
  #IPython.display.clear_output()
  val_performance['LSTM'] = lstm_model["model"].evaluate(wide_window.val, return_dict=True)
  performance['LSTM'] = lstm_model["model"].evaluate(wide_window.test, verbose=0, return_dict=True)
  
  wide_window.plot(models = [ lstm_model ], max_subplots = max_subplots, title=lstm_model["title_long"])
  
  if draw_bias:
    wide_window.plot(models = [ lstm_model ], max_subplots = max_subplots, draw_bias = draw_bias, title=lstm_model["title_long"])



if L_mlst_recurrent_neural_network:
  multi_lstm_model = {
    "title_long":  "Multi step recurrent neural network (LSTM)",
    "title_short": "mlst_lstm",
    "model": tf.keras.Sequential([
      # Shape [batch, time, features] => [batch, lstm_units].
      # Adding more `lstm_units` just overfits more quickly.
      tf.keras.layers.LSTM(32, return_sequences=False),
      # Shape => [batch, out_steps*features].
      tf.keras.layers.Dense(OUT_STEPS*num_features, kernel_initializer=tf.initializers.zeros()),
      # Shape => [batch, out_steps, features].
      tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])
  }
  print(); print (multi_lstm_model["title_long"]); print()
  
  history = stuff.compile_and_fit(multi_lstm_model["model"], multi_window, MAX_EPOCHS=MAX_EPOCHS)
  
  multi_lstm_model["history"] = history
  
  multi_val_performance['LSTM'] = multi_lstm_model["model"].evaluate(multi_window.val, return_dict=True)
  multi_performance['LSTM'] = multi_lstm_model["model"].evaluate(multi_window.test, verbose=0, return_dict=True)
  
  multi_window_models.append( multi_lstm_model )
  
  if single_plot:
    multi_window.plot(models = [ multi_lstm_model ], max_subplots = max_subplots, title=multi_lstm_model["title_long"])


if L_mlst_2level_recurrent_neural_network:
  multi_2llstm_model = {
    "title_long":  "Multi step 2-layer recurrent neural network (LSTM)",
    "title_short": "mlst_2l_lstm",
    "model": tf.keras.Sequential([
      # Shape [batch, time, features] => [batch, lstm_units].
      # Adding more `lstm_units` just overfits more quickly.
      tf.keras.layers.LSTM(32, return_sequences=True),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.LSTM(32, return_sequences=False),
      tf.keras.layers.Dropout(0.2),
      # Shape => [batch, out_steps*features].
      tf.keras.layers.Dense(OUT_STEPS*num_features, kernel_initializer=tf.initializers.zeros()),
      # Shape => [batch, out_steps, features].
      tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])
  }
  print(); print (multi_2llstm_model["title_long"]); print()
  
  history = stuff.compile_and_fit(multi_2llstm_model["model"], multi_window, MAX_EPOCHS=MAX_EPOCHS)
  
  multi_2llstm_model["history"] = history
  
  multi_val_performance['2lLSTM'] = multi_2llstm_model["model"].evaluate(multi_window.val, return_dict=True)
  multi_performance['2lLSTM'] = multi_2llstm_model["model"].evaluate(multi_window.test, verbose=0, return_dict=True)
  
  multi_window_models.append( multi_2llstm_model )
  
  if single_plot:
    multi_window.plot(models = [ multi_2llstm_model ], max_subplots = max_subplots, title=multi_2llstm_model["title_long"])



if L_mlst_autoregressive_recurrent_neural_network:
  
  feedback_model = {
    "title_long":  "Multi step autoregressive recurrent neural network (LSTM)",
    "title_short": "mlst_ar_lstm",
    "model": ar_lstm.FeedBack(units=32, out_steps=OUT_STEPS, num_features = num_features)
  }
  
  prediction, state = feedback_model["model"].warmup(multi_window.example[0])

  print(); print (feedback_model["title_long"]); print()
  
  history = stuff.compile_and_fit(feedback_model["model"], multi_window, MAX_EPOCHS=MAX_EPOCHS)
  
  feedback_model["history"] = history
  
  multi_val_performance['AR LSTM'] = feedback_model["model"].evaluate(multi_window.val, return_dict=True)
  multi_performance['AR LSTM'] = feedback_model["model"].evaluate(multi_window.test, verbose=0, return_dict=True)
  
  multi_window_models.append( feedback_model )
  
  if single_plot:
    multi_window.plot(models = [ feedback_model ], max_subplots = max_subplots, title=feedback_model["title_long"])


if L_mlst_gated_recurrent_units:
  multi_gru_model = {
    "title_long":  "Multi step gated recurrent units (GRU)",
    "title_short": "mlst_gru",
    "model": tf.keras.Sequential([
      tf.keras.layers.GRU(256, return_sequences=False),
      tf.keras.layers.Dense(OUT_STEPS*num_features, kernel_initializer=tf.initializers.zeros()),
      tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])
  }
  print(); print (multi_gru_model["title_long"]); print()
  
  history = stuff.compile_and_fit(multi_gru_model["model"], multi_window, MAX_EPOCHS=MAX_EPOCHS)
  
  multi_gru_model["history"] = history
  
  multi_val_performance['GRU'] = multi_gru_model["model"].evaluate(multi_window.val, return_dict=True)
  multi_performance['GRU'] = multi_gru_model["model"].evaluate(multi_window.test, verbose=0, return_dict=True)
  
  multi_window_models.append( multi_gru_model )
  
  if single_plot:
    multi_window.plot(models = [ multi_gru_model ], max_subplots = max_subplots, title=multi_gru_model["title_long"])
  

if L_mlst_bidirectional_lstm:
  multi_bilstm_model = {
    "title_long":  "Multi step Bidirectional recurrent neural network (biLSTM)",
    "title_short": "mlst_bilstm",
    "model": tf.keras.Sequential([
      tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=False)),
      tf.keras.layers.Dense(OUT_STEPS*num_features, kernel_initializer=tf.initializers.zeros()),
      tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])
  }
  print(); print (multi_bilstm_model["title_long"]); print()
  
  history = stuff.compile_and_fit(multi_bilstm_model["model"], multi_window, MAX_EPOCHS=MAX_EPOCHS)
  
  multi_bilstm_model["history"] = history
  
  multi_val_performance['biLSTM'] = multi_bilstm_model["model"].evaluate(multi_window.val, return_dict=True)
  multi_performance['biLSTM'] = multi_bilstm_model["model"].evaluate(multi_window.test, verbose=0, return_dict=True)
  
  multi_window_models.append( multi_bilstm_model )
  
  if single_plot:
    multi_window.plot(models = [ multi_bilstm_model ], max_subplots = max_subplots, title=multi_bilstm_model["title_long"])


if L_mlst_transformer:
  print (train_df)
  print (multi_window.test)
  input_shape = train_df.shape[1:]
  print (input_shape)
  
  for example_inputs, example_labels in multi_window.train.take(1):
    print (single_step_window.train.take(1))
    print ("++")
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')
  
  input_shape = example_inputs.shape
  print (input_shape)
  
  multi_transformer_model = {
    "title_long":  "Multi step transformer",
    "title_short": "mlst_transformer",
    "model": transformer.build_model(
      input_shape,
      head_size=256,
      num_heads=4,
      ff_dim=4,
      num_transformer_blocks=4,
      mlp_units=[128],
      mlp_dropout=0.4,
      dropout=0.25,
  )}
  print(); print (multi_transformer_model["title_long"]); print()
  
  if False:
    multi_transformer_model["model"].compile( loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=1e-4), metrics=["sparse_categorical_accuracy"])
    
    callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
    
    model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=150,
        batch_size=64,
        callbacks=callbacks,
    )
    
    model.evaluate(x_test, y_test, verbose=1)
  
  history = stuff.compile_and_fit(multi_transformer_model["model"], multi_window, MAX_EPOCHS=MAX_EPOCHS)
  
  multi_transformer_model["history"] = history
  
  multi_val_performance['biLSTM'] = multi_transformer_model["model"].evaluate(multi_window.val, return_dict=True)
  multi_performance['biLSTM'] = multi_transformer_model["model"].evaluate(multi_window.test, verbose=0, return_dict=True)
  
  multi_window_models.append( multi_transformer_model )
  
  if single_plot:
    multi_window.plot(models = [ multi_transformer_model ], max_subplots = max_subplots, title=multi_transformer_model["title_long"])



if len(multi_window_models) > 0:
  multi_window.plot(models = multi_window_models, max_subplots = max_subplots)
  
  plt.figure(figsize=(12, 8))
  plt.title("loss ")
  plt.yscale("log")
  for m in range(len(multi_window_models)):
    plt.plot(multi_window_models[m]["history"].history["loss"], label = multi_window_models[m]["title_short"], linestyle='-')
    plt.legend()
    plt.xlabel('epochs')


x = np.arange(len(multi_performance))
width = 0.3

print (multi_performance)
print (multi_performance.values())

metric_name = 'mean_absolute_error'
val_mae = [v[metric_name] for v in multi_val_performance.values()]
test_mae = [v[metric_name] for v in multi_performance.values()]

plt.figure(figsize=(12, 8))
plt.title("MAE ")
plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=multi_performance.keys(), rotation=45)
plt.ylabel('MAE (average over all outputs)')
plt.legend()



#IPython.display.clear_output()

plt.show()




exit()


