# Libraries
import os
import numpy as np
from numpy import load, save
import time
import sys

# External libs
import utils_pre
from utils_pre import *

import model
from model import *

import utils_deep
from utils_deep import *

# import validation_tools
# from validation_tools import *

# import CMC_tools
# from CMC_tools import compute_CMC_scores


# READ DATA
dataset=sys.argv[-1]
n_iter = int(sys.argv[-2]) # 10000 No. of training iterations
embeddingsize = int(sys.argv[-3]) #32
semihard_no = int(sys.argv[-4])
hard_no = 0
easy_no = 0
random_no = 0
batch_no=int(sys.argv[-5])

print('The code is running for database: %s, iteration: %s, embedded size: %s, semi-hard batch size: %s, overal batch size: %s'%(dataset, n_iter, embeddingsize,semihard_no,batch_no), flush= True)
path = '/home/nasrim/data/'+dataset+'/'   # Dont forget the last backslash
x1_list = load(path+'x1_list_300.npy')
x2_list = load(path+'x2_list_300.npy')
y_list = load(path+'y1_list_300.npy')
traj_id = load(path+'y3_list_300.npy')

# HYPER PARAMETERS

evaluate_every = 100 # interval for evaluating on one-shot tasks
n_val = 250 # how many one-shot tasks to validate on
n_filters=50
kernel_size=10
strides=1
# pool_size=5
n_units=[50, 1]


n_iteration=0
path_result = '/home/nasrim/data/results/'+dataset+'/'+str(embeddingsize)+'/'+str(batch_no)+str(hard_no)+str(semihard_no)+str(easy_no)+str(random_no)+'/'
if not os.path.exists(path_result):
    os.makedirs(path_result)


# Data preprocessing
percent = 0.8

# Shuffle the input data
x1_list_s, x2_list_s, y_list_s = shuffle_data(x1_list, x2_list, traj_id)

i = 90000
x1_list_s = x1_list_s [:i,:,:]
x2_list_s= x2_list_s[:i,:,:]
y_list_s= y_list_s[:i]

# Make train and test
x1_test, x2_test, x1_train, x2_train, y_train, y_test = split_data(x1_list_s, x2_list_s, y_list_s, percent)

# Combine x1 and x2 -> useful fro triplet. loss function
x_train, y_train = combi_classed (x1_train, x2_train, y_train)
x_test, y_test = combi_classed (x1_test, x2_test, y_test)

# Make Dataset ready
nb_classes = np.unique(traj_id)
img_rows, img_cols = x1_list[:,:,1:3].shape[1], x1_list[:,:,1:3].shape[2]
input_shape = (img_rows, img_cols)
dataset_train,dataset_test,x_train_origin,y_train_origin,x_test_origin,y_test_origin = buildDataSet(x_train[:,:,1:3], y_train[:], x_test[:,:,1:3], y_test[:], img_rows, img_cols,nb_classes)
nb_classes_train = np.unique(y_train_origin)
nb_classes_test = np.unique(y_test_origin)

print("Checking shapes for class 0 (train) : ",dataset_train[0].shape)
print("Checking shapes for class 0 (test) : ",dataset_test[0].shape)

save(path_result+'x_test_origin.npy', x_test_origin)
save(path_result+'x_train_origin.npy', x_train_origin)
save(path_result+'dataset_test.npy', dataset_test)
save(path_result+'dataset_train.npy', dataset_train)
save(path_result+'nb_classes_test.npy', nb_classes_test)
save(path_result+'nb_classes_train.npy', nb_classes_train)


# Make Model

network = build_network(input_shape, embeddingsize, n_filters, kernel_size,strides,n_units)
network_train = build_model(input_shape,network)
optimizer = Adam(lr = 1e-4)
network_train.compile(loss=None,optimizer=optimizer)
network.summary()

# Train the model

myfile = open('training_info.txt', 'w')
loss_list=[]

print("Starting training process!", flush=True)
print("-------------------------------------", flush=True)
t_start = time.time()
for i in range(1, n_iter+1):
    triplets = get_batch_hard(batch_no,hard_no,semihard_no,easy_no,random_no,network, dataset_train, nb_classes_train,0.2)
    # print(i)
    loss = network_train.train_on_batch(triplets, None)
    
    # Save results
    model_yaml = network.to_yaml()
    with open(path_result+"model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    # serialize weights to HDF5
    network.save_weights(path_result+"model.h5")
    #print("Saved model to disk", flush=True)

    n_iteration += 1
    loss_list.append(loss)
    save(path_result+'loss.npy', loss_list)
    if i % evaluate_every == 0:
        print("\n ------------- \n")
        print("[{3}] Time for {0} iterations: {1:.1f} mins, Train Loss: {2}".format(i, (time.time()-t_start)/60.0,loss,n_iteration), flush=True)
        # probs,yprob = compute_probs(network,x_test_origin[:n_val,:,:],y_test_origin[:n_val])
        # fpr, tpr, thresholds,auc = compute_metrics(probs,yprob)
        #draw_roc(fpr, tpr,thresholds)

myfile.close()

# Save the results

model_yaml = network.to_yaml()
with open(path_result+"model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
network.save_weights(path_result+"model.h5")
print("Saved model to disk", flush=True)

# Save output files
save(path_result+'x_test_origin.npy', x_test_origin)
save(path_result+'x_train_origin.npy', x_train_origin)
save(path_result+'dataset_test.npy', dataset_test)
save(path_result+'dataset_train.npy', dataset_train)
save(path_result+'nb_classes_test.npy', nb_classes_test)
save(path_result+'nb_classes_train.npy', nb_classes_train)
save(path_result+'loss.npy', loss_list)


