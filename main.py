import os
import sys
sys.path.append(os.getcwd() + r'\deepbeliefpack')

import pickle
import dataload as dld
from torchvision import transforms
import visual as vs
import rbm as rbms
import dbn as dbns

import torch
# torch.set_default_tensor_type('torch.cuda.FloatTensor')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available(), device)


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
    batch_size = 100
    transfs = [transforms.ToTensor()]
    loader = dld.LoadDataset(batch_size, transfs)
    Xtrain, Xtest, Ytrain, Ytest = loader.yield_tensor_data()
    vs.plot_images_grid(Xtrain[0,:,:],Ytrain[0,:,:], title = 'Raw data')
    
    dataset = {'train' : [Xtrain,Ytrain],
               'test'  : [Xtest, Ytest]}
    with open('dataset\dataset.pickle', 'wb') as handle: pickle.dump(dataset,handle)
else:
    with open('dataset\dataset.pickle','rb') as handle : dataset = pickle.load(handle)
    [Xtrain,Ytrain] = dataset['train']
    [Xtest, Ytest]  = dataset['test']
#end

visible_dim = Xtrain[0,0].shape[0]
hidden_dims = [500, 500, 2000]

epochs        = 10
learning_rate = 0.01
weights_decay = 0.0005
momentum      = 0.5
mcmc_steps    = 1

hyperparameters = {'epochs'        : epochs,
                   'learning_rate' : learning_rate,
                   'weights_decay' : weights_decay,
                   'momentum'      : momentum,
                   'mcmc_steps'    : mcmc_steps}

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
        net_model = {'model' : rbm, 'hyperparameters' : hyperparameters}
        with open('models\rbm.pickle','rb') as handle: rbm = pickle.load(handle)
    #end
    
elif model == 'dbn':
    
    if train:
        dbn = dbns.DeepBeliefNet(visible_dim, hidden_dims)
        [rbm.cuda() for rbm in dbn.rbm_layers]
        dbn.contrastive_divergence_train([Xtrain,Ytrain], epochs,
                                         learning_rate, weights_decay, momentum,
                                         mcmc_steps)
        dbn.reconstruction([Xtest,Ytest], title = 'Original samples')
        dbn.corrupt_and_recreate(Xtest.clone(), Ytest.clone())
        dbn.noise_and_denoise(Xtest.clone(), Ytest.clone(), title = 'Noised samples')
        if save:
            net_model = {'model' : dbn, 'hyperparameters' : hyperparameters}
            with open('models\dbn.pickle','wb') as handle: pickle.dump(net_model, handle)
        #end
    elif load:
        with open('models\dbn.pickle','rb') as handle: dbn = pickle.load(handle)
    #end
#end


