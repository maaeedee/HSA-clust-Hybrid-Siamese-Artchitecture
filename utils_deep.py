import numpy as np

def get_batch_random(batch_size, dataset, nb_classes):
    """
    Create batch of APN triplets with a complete random strategy

    Arguments:
    batch_size -- integer

    Returns:
    triplets -- list containing 3 tensors A,P,N of shape (batch_size,w,h,c)
    """

    X = dataset

    m, w, h = X[0].shape  # c removed


    # initialize result
    triplets=[np.zeros((batch_size,w, h)) for i in range(3)]   # c removed
    # print(X[0].shape)

    for i in range(batch_size):
        #Pick one random class for anchor
        anchor_class = np.random.randint(0, len(nb_classes))
        nb_sample_available_for_class_AP = X[anchor_class].shape[0]

        #Pick two different random pics for this class => A and P
        [idx_A,idx_P] = np.random.choice(nb_sample_available_for_class_AP,size=2,replace=False)

        #Pick another class for N, different from anchor_class
        negative_class = (anchor_class + np.random.randint(1,len(nb_classes))) % len(nb_classes)
        nb_sample_available_for_class_N = X[negative_class].shape[0]

        #Pick a random pic for this negative class => N
        idx_N = np.random.randint(0, nb_sample_available_for_class_N)


        triplets[0][i,:,:] = X[anchor_class][idx_A,:,:]
        triplets[1][i,:,:] = X[anchor_class][idx_P,:,:]
        triplets[2][i,:,:] = X[negative_class][idx_N,:,:]

    return triplets

def drawTriplets(tripletbatch, nbmax=None):
    """display the three images for each triplets in the batch
    """
    labels = ["Anchor", "Positive", "Negative"]

    if (nbmax==None):
        nbrows = tripletbatch[0].shape[0]
    else:
	nbrows = min(nbmax,tripletbatch[0].shape[0])

    for row in range(nbrows):
        fig=plt.figure(figsize=(16,2))

        for i in range(3):
            subplot = fig.add_subplot(1,3,i+1)
            axis("off")
            #plt.imshow(tripletbatch[i][row,:,:],vmin=0, vmax=1,cmap='Greys')
            #print([j for j in tripletbatch[i][row,:,0:]])
            plt.scatter([j for j in tripletbatch[i][row,:,0] if j!=0], [j for j in tripletbatch[i][row,:,1] if j!=0], c = 'black')
            plt.plot([j for j in tripletbatch[i][row,:,0] if j!=0], [j for j in tripletbatch[i][row,:,1] if j!=0], c = 'blue')
            subplot.title.set_text(labels[i])

def compute_dist(a,b):
  return np.sqrt(np.sum(np.square(a-b)))

def get_batch_hard(draw_batch_size,hard_batchs_size,semihard_batchs_size,easy_batchs_size,norm_batchs_size,network,dataset,nb_classes, margin):
    """
    Create batch of APN "hard" triplets

    Arguments:
    draw_batch_size -- integer : number of initial randomly taken samples
    hard_batchs_size -- interger : select the number of hardest samples to keep
    norm_batchs_size -- interger : number of random samples to add

    Returns:
    triplets -- list containing 3 tensors A,P,N of shape (hard_batchs_size+norm_batchs_size,w,h,c)
    """


    X = dataset

    m, w, h = X[0].shape  # c removed


    #Step 1 : pick a random batch to study
    studybatch = get_batch_random(draw_batch_size,dataset, nb_classes)

    #Step 2 : compute the loss with current network : d(A,P)-d(A,N). The alpha parameter here is omited here since we want only to order them
    studybatchloss = np.zeros((draw_batch_size))

    #Compute embeddings for anchors, positive and negatives
    #print('*',studybatch[0][:,:,:].shape)
    A = network.predict(studybatch[0][:,:,:])
    P = network.predict(studybatch[1][:,:,:])
    N = network.predict(studybatch[2][:,:,:])

    #Compute d(A,P)-d(A,N) # HARD
    studybatchloss = np.sqrt(np.sum(np.square(A-P),axis=1)) - np.sqrt(np.sum(np.square(A-N),axis=1))

    #Sort by distance (high distance first) and take the hardest
    selection = np.argsort(studybatchloss)[::-1][:hard_batchs_size]

    #Compute d(A,N)-d(A,P) # EASY
    studybatchloss = -np.sqrt(np.sum(np.square(A-P),axis=1)) + np.sqrt(np.sum(np.square(A-N),axis=1))
    #Sort by distance (high distance first) and take the EASIEST
    selection1 = np.argsort(studybatchloss)[::-1][:easy_batchs_size] #


    #Compute d(A,N)-d(A,P) SEMI-HARD
    semihard_index1 = np.squeeze(np.where(np.sqrt(np.sum(np.square(A-P),axis=1)) + margin > np.sqrt(np.sum(np.square(A-N),axis=1))))
    semihard_index2 = np.squeeze(np.where(np.sqrt(np.sum(np.square(A-P),axis=1)) < np.sqrt(np.sum(np.square(A-N),axis=1))))
    semihard_index = np.intersect1d(semihard_index1,semihard_index2)

    selection2 = semihard_index[:semihard_batchs_size] #

    selection = np.append(selection,selection1) #Hard & Easy
    selection = np.append(selection,selection2) #Hard & Easy & SemiHard

    #Draw other random samples from the batch
    selection2 = np.random.choice(np.delete(np.arange(draw_batch_size),selection),norm_batchs_size,replace=False)

    selection = np.append(selection,selection2) #Hard & Easy & SemiHard & Random

    triplets = [studybatch[0][selection,:,:], studybatch[1][selection,:,:], studybatch[2][selection,:,:]]

    return triplets

