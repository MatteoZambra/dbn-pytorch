

import os
import sys
sys.path.append(os.getcwd() + r'\deepbeliefpack')

import pickle
import dataload as dld
from torchvision import transforms
import visual as vs
import rbm as rbms
import dbn as dbns


"""
The main program is instructed whether to train a model again, which model, whether
to save the dataset and model as serialized objects.

Hyper-paramters as batch size, learning rate and network architecture must be 
defined by the user.
"""

data  = False
train = True
save  = False
load  = False
model = 'dbn'

if data:
    batch_size = 128
    transfs = [transforms.ToTensor()]
    loader = dld.LoadDataset(batch_size, transfs)
    Xtrain, Ytrain, Xtest, Ytest = loader.yield_data(binarize = True, factor = 4)
    vs.plot_images_grid(Xtrain[0],Ytrain[0], title = 'Raw data')
    
    dataset = {'train' : [Xtrain,Ytrain],
               'test'  : [Xtest, Ytest]}
    with open('dataset\dataset.pickle', 'wb') as handle: pickle.dump(dataset,handle)
else:
    with open('dataset\dataset.pickle','rb') as handle : dataset = pickle.load(handle)
    [Xtrain,Ytrain] = dataset['train']
    [Xtest, Ytest]  = dataset['test']
#end

visible_dim = Xtrain[0][0].shape[0]
hidden_dims = [100, 200]

epochs        = 10
learning_rate = 0.025
weights_decay = 0.0001
momentum      = 0.5
mcmc_steps    = 2


if model == 'rbm':
    
    if train:
        rbm = rbms.RestrictedBoltzmannMachine(visible_dim, hidden_dims[0])
        rbm.contrastive_divergence_train([Xtrain,Ytrain], epochs,
                                          learning_rate, weights_decay, momentum,
                                          mcmc_steps)
        rbm.reconstruction([Xtest,Ytest])
        if save:
            with open('models\rbm.pickle','wb') as handle: pickle.dump(rbm, handle)
        #end
        
    elif load:
        with open('models\rbm.pickle','rb') as handle: rbm = pickle.load(handle)
    #end
    
elif model == 'dbn':
    
    if train:
        dbn = dbns.DeepBeliefNet(visible_dim, hidden_dims)
        dbn.contrastive_divergence_train([Xtrain,Ytrain], epochs,
                                         learning_rate, weights_decay, momentum,
                                         mcmc_steps)
        dbn.reconstruction([Xtest,Ytest])
        if save: 
            with open('models\dbn.pickle','wb') as handle: pickle.dump(dbn, handle)
        #end
    elif load:
        with open('models\dbn.pickle','rb') as handle: dbn = pickle.load(handle)
    #end
#end

#%%
# import numpy as np

# side_dim     = int(np.sqrt(Xtrain[0][0].shape[0]))
# row          = 3
# rows_to_kill = 5

# """
# Corruption.
# The images are vectorized, i.e. 28x28 matrices flattened. 
# Recall the good old row-major order. Assume we want to delete the i to i+k-th 
# rows, setting all the pixels is such rows to 0.0 -- i.e. black. 
# Then all the matrix entries from (i+1)*28 to (i+k)*dim + dim should be selected.
# Considering the flattened matrix, it translated simply to setting to 0.0 the 
# entries of the vector X[(i+1)*dim : (i+k)*dim +dim].
# """

# x = Xtest[0] # one batch
# for x in Xtest:
#     x[:, (row + 1)*side_dim : (row + rows_to_kill)*side_dim + side_dim] = 0.0
# #end
# vs.plot_images_grid(Xtest[0], Ytest[0], title = 'corrupted images')

# dbn.reconstruction([Xtest,Ytest])




