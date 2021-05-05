# HSA-clust-Hybrid-Siamese-Artchitecture
This repository includes the source code of the following paper.

The hybrid siamese architecture employes the triplet loss function to retrieve the embeddings of the raw GPS trajectories. Such embeddings can be used for user trajectory identification and verficiation and clustering the trajectories into the groups of people who have same movemenet behaviour. 
This model is particularely designed and tested to capture similar trajectories in the limited area such as school playground, nursing homes and sport clubs. 

----------
There are following files in this repository:
* `data_augmentation.py`: Generates the augmented GPS trajectory from raw GPS trajectories
* `utils_pre.py`: Includes functions which is required to organize the data into the proper shape and type.
* `utils_deep.py`: Includes functions which is used to create triplets for the deep model.
* `model.py`: Builds the model which is designed and trained in this study.
* `main.py`: Includes the main implementation of the project.

In order to run the model you need to take the following steps:
#### 1. Run the `data_augmentation.py`, to generate augmented trajectories that you need for training the model.
#### 2. Run the `main.py` to train the model via below command:
`python main.py <batch_size> <semi-hardbatchsize> <embeddingsize> <#iteration> 'dataset'`
