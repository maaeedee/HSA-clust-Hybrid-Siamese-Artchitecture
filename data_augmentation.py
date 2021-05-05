# -*- coding: utf-8 -*-
"""data_augmentation.py

# Notebook: Generate Training Dataset

In this Notebook, we want to simulate training dataset from the real world dataset. There are two steps in making such data:
* 1) Create pair of trajectories from the original set
* 2) Create label per pair of trajectories

# Required packages
"""

import numpy as np
from numpy import save, load

import pandas as pd

import matplotlib.pyplot as plt

import glob

"""# Define functions"""

def read_data(path, format, n, min_len):

  data_list=[]
  c=0
  for file in glob.glob(path+format):
    #print(file)
    try:
      df = pd.read_csv(file,  header=None, sep=',|;')

    except:
      df = pd.DataFrame ()


    if ((len(df)>min_len)):
      data_list.append(df)
      c = c + 1
    if c >= n:
      break

  dataset=pd.concat(data_list)
  dataset.columns=['id', 'date', 'lng', 'lat']
  dataset=dataset.reset_index(drop=True)
  print ('Step1: Read files')
  return dataset

def linux_TS(df):

  df.date=pd.to_datetime(df.date)
  df['TS'] = (pd.DatetimeIndex(df.date-pd.Timedelta('02:00:00')).astype(np.int64) // 10**9)

  return df

def min_drop_func(df, min_len):

  data_list=[]
  ids=np.unique(df.id)
  for i in ids:
    temp=df[df.id==i]
    if len (temp) > min_len:
      data_list.append(temp)
  data_list = pd.concat(data_list)
  data_list=data_list.reset_index(drop=True)

  return data_list

def data_window(df, win_size, i):

  temp=df.loc[i:i+win_size,].copy()

  return temp

def gap_finder (df, max_gap):

  temp = df.sort_values('TS')
  dif_time = np.diff(temp.TS)
  max_val = np.max(dif_time)
  #print(max_val)

  if max_val < max_gap:
    #print("*****")
    gap_flag = 1

  else:
    gap_flag = 0
    print ('There is a gap, excluded from data...')


  return gap_flag, temp

def drop_indx (df_d, r):

    drop_indices = np.random.choice(df_d.index, int(r*len(df_d)), replace=False)
    temp_d=df_d.copy()
    temp_d.loc[drop_indices,['lng_1','lat_1']]=0

    return temp_d

def add_offset (df):
  var_lng=np.var(df.lng_1)
  var_lat=np.var(df.lat_1)

  # Set a length of the list to length of dataset
  noise_lng=[var_lng]*len(df)
  noise_lat=[var_lat]*len(df)

  # print (noise_lat)

  df['lng_1'] = df.lng_1 + noise_lng
  df['lat_1'] = df.lat_1 + noise_lat

  return df, var_lng, var_lat

def second_traj(df1, df2):

  df1_list = np.array(df1[['id','lng','lat']])
  df2_list = np.array(df2[['id','lng_1','lat_1']])

  return df1_list, df2_list

def label_maker (var_lng, var_lat, r):

  label = r*(var_lng+var_lat)

  return label

def shift_time(df, s):
  df['lng_1']= df.lng.shift(s)
  df['lat_1']= df.lat.shift(s)

  return df

def make_pairs(org_df, win_size):
  print('Start making trajectory pairs ...')

  y_list=[]
  x1_list=[]
  x2_list=[]
  r=[0.1,0.2,0.4,0.6, 0.7, 0.8]
  shifts=[-2, -1, 0, 1, 2]

  ids = np.unique(org_df.id)

  # Extract IDs
  for idd in ids:
    temp=org_df[org_df.id==idd]
    temp=temp.reset_index(drop=True)
    # Extract Windows in IDs
    for i in range(0,len(temp)-win_size,win_size+1):
      for ri in r:
        for s in shifts:

          # make window of time
          temp_0 = data_window(temp, win_size, i)
          # check the gap
          gap_flag, tempx = gap_finder (temp_0, max_gap)

          if gap_flag:
            # Create 2nd trajectory

            # Shift in time
            temps = shift_time(tempx, s)

            # Add offset
            temp_1, var_lng, var_lat = add_offset (temps)

            # Drop index
            temp_2 = drop_indx (temp_1, ri)

            #print(temp_2)
            # Create Trajectory pair
            df1_list, df2_list = second_traj(temp_0, temp_2)
            x1_list.append(df1_list)
            x2_list.append(df2_list)

            # Create Label
            label = label_maker(var_lng, var_lat, ri)
            y_list.append(label)


  x1_list = np.array(x1_list, dtype=float)
  x2_list = np.array(x2_list, dtype=float)
  y_list = np.array(y_list, dtype=float)

  print ('Step2: Trajectory pair is created.')
  return x1_list, x2_list, y_list

def plot_example( x1_list, x2_list, n):

  fig, axis = plt.subplots(figsize = (10, 10))
  plt.plot([i[1] for i in x1_list[n]], [i[2] for i in x1_list[n] ], label='Original trajectory')
  plt.plot([i[1] for i in x2_list[n] if i[1]!=0], [i[2] for i in x2_list[n] if i[2]!=0], label='Trajectory pair')
  for indx, i in enumerate(x2_list[n]):
    if i[1] == 0:

      plt.scatter(x1_list[n][indx][1], x1_list[n][indx][2], c='red')

  plt.scatter(x1_list[n][indx][1], x1_list[n][indx][2], c='red', label='Dropped points')

  # Make legend
  plt.legend(loc=4)

  # Hide values on X and Y axis
  plt.xticks([])
  plt.yticks([])

  # Set labeld for axis
  plt.xlabel('Longitude')
  plt.ylabel('Latitude')

  plt.title ('Simulating trajecory pair')

  return 0

"""# Processing the data"""

# Initial variables
path='/home/nasrim/data/taxi_log_2008_by_id'
format='*.txt'
min_len=3
win_size=29
max_gap=1000000000

# Functions
# Read data files
dataset_0 = read_data(path, format, n=5, min_len=3)

# Drop short trajectories
dataset_1 = min_drop_func(dataset_0, min_len)

# Add Linux Timestamp
dataset_2 = linux_TS (dataset_1)

# Make pairs and so on
x1_list, x2_list, y_list = make_pairs(dataset_2, win_size)

# Save datafiles
save('x1_list.npy',x1_list)
save('x2_list.npy',x2_list)
save('y_list.npy',y_list)

# Plot an example, uncomment if you want to see how the 226th sample is look like.
# plot_example( x1_list, x2_list, n=226)




