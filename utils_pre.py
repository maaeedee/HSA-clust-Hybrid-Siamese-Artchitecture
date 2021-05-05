# libraries
import numpy as np
from random import shuffle, randrange

# Split the data
def split_data(x1_list_n, x2_list_n, y_list_n, percent):

    split_value=int(x1_list_n.shape[0]*percent)

    x1_test=x1_list_n[split_value:]
    x2_test=x2_list_n[split_value:]
    y_test=y_list_n[split_value:]

    x1_train=x1_list_n[:split_value]
    x2_train=x2_list_n[:split_value]
    y_train=y_list_n[:split_value]

    return x1_test, x2_test, x1_train, x2_train, y_train, y_test

# Shuffle Data
def shuffle_data(x1_list, x2_list, y_list):

  N = x1_list.shape[0]
  ind_list = [i for i in range(N)]
  shuffle(ind_list)

  x1_list_n  = x1_list[ind_list, :,:]
  x2_list_n  = x2_list[ind_list, :,:]
  y_list_n = y_list[ind_list,]


  return x1_list_n, x2_list_n, y_list_n

# Combine classes
def combi_classed (x1, x2, y):
  # Make training and labeling data ready
  data_list = []
  #x1_list, x2_list = norm_data (x1_list, x2_list)


  for i in np.unique(y):
    inx = np.where(y == i)[0][0]
    data_list.append(x1[inx])

  x_list = np.array(data_list)
  x_list = np.concatenate((x2,x_list))
  y_list = np.concatenate((y, np.unique(y)))

  return x_list, y_list

# Normalizing the data
def norm_data (x):

  x[:,1] = (x[:,1]- np.mean(x[:,1]))/np.std(x[:,1])
  x[:,0] = (x[:,0]- np.mean(x[:,0]))/np.std(x[:,0])

  return x

# Draw Triplets
def DrawPics(tensor,nb=0,template='{}',classnumber=None):
    if (nb==0):
        N = tensor.shape[0]
    else:
	N = min(nb,tensor.shape[0])
    fig=plt.figure(figsize=(16,2))
    nbligne = floor(N/20)+1
    for m in range(N):
        subplot = fig.add_subplot(nbligne,min(N,20),m+1)
        axis("off")
        plt.plot([i for i in tensor[m,:,0] if i!=0], [i for i in tensor[m,:,1] if i!=0])
        if (classnumber!=None):
            subplot.title.set_text((template.format(classnumber)))

# Build dataset
def buildDataSet(x_train_origin, y_train_origin, x_test_origin, y_test_origin,img_rows, img_cols,nb_classes):
  
    x_train_origin = x_train_origin.reshape(x_train_origin.shape[0], img_rows, img_cols )
    x_test_origin = x_test_origin.reshape(x_test_origin.shape[0], img_rows, img_cols)

    dataset_train = []
    dataset_test = []

    #Sorting trajectories by classes and normalize values 0=>1
    for n in nb_classes:

      try:

	trj_class_n = np.asarray([row for idx,row in enumerate(x_train_origin) if y_train_origin[idx]==n])
        # dataset_train.append(norm_data (trj_class_n))
        trj_class_n[:,:,0] = trj_class_n[:,:,0]/180
        trj_class_n[:,:,1] = trj_class_n[:,:,1]/90
        dataset_train.append(trj_class_n)

      except:
	continue

      try:

	trj_class_n = np.asarray([row for idx,row in enumerate(x_test_origin) if y_test_origin[idx]==n])
        # dataset_test.append(norm_data (trj_class_n))
        trj_class_n[:,:,0] = trj_class_n[:,:,0]/180
        trj_class_n[:,:,1] = trj_class_n[:,:,1]/90
        dataset_test.append(trj_class_n)

      except:
	continue

    return dataset_train,dataset_test,x_train_origin,y_train_origin,x_test_origin,y_test_origin
