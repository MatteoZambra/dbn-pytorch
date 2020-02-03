
"""
Class Deep Belief Network.
Provides the nn.Module subclass containing all what needed to 
train such a device.

The DBN here implemented is thought to be trained relying on the greedy layer-wise
algorithm presented by Hinton et al (2006). 

For an extensive account on the subject see
~ Hinton, Osindero and Teh (2006) A Fast Learning Algorithm for Deep Belief Nets

Also an useful reference for the implementation is the Appendix of 
~ Bengio, Lamblin Popovici and Larochelle (2007) Greedy Layer-wise training
                                                 of Deep Networks
in particular pseudocodes in Algorithms 1 and 2
"""

import copy
import numpy as np
import torch
import torch.nn as nn

import rbm
import visual as vs


torch.manual_seed(618)


class DeepBeliefNet(nn.Module):
    
    def __init__(self, visible_dim, hidden_dims):
        """
        Constructor.
        
        The attributes of this module are the number of RBM building blocks and
        the list of such layers. Each of such layers has its parameters saved 
        as nn.Parameters. In this scope of RBMs it is not strictly necessary
        since the nn.Parameters are used to keep the gradients wrt such values.
        
        Input:
            ~ visible_dim (integer) : number of visible units
            ~ hidden_dims (list of integers) : units for each hidden layer
            
        Returns:
            ~ nothing
        """
        
        self.num_rbms_layers = len(hidden_dims)
        self.rbm_layers = []
        
        self.rbm_layers.append(rbm.RestrictedBoltzmannMachine(visible_dim, hidden_dims[0]))
        for i in range(self.num_rbms_layers-1):
            self.rbm_layers.append(rbm.RestrictedBoltzmannMachine(hidden_dims[i], hidden_dims[i+1]))
        #end
    #end
    
    
    def forward(self, visible):
        """
        Again as in RBM, the forward method is thought to peform bottom-up and
        top-down pass, in order to reconstruct a given data sample.
        
        Input:
            ~ visible (torch.Tensor) : as usual
            
        Returns:
            ~ visible_reconstruction (torch.Tensor) : again
        """
        
        _reconstr = visible.clone()
        for i in range(len(self.rbm_layers)):
            print('Layer {}'.format(i+1))
            _reconstr = self.rbm_layers[i].v_to_h(_reconstr)
            # _reconstr = torch.bernoulli(torch.sigmoid(_reconstr))
        #end
        
        """
        Gibbs sampling in the top layer RBM to get unbiased samples from the 
        model joint distribution (see Appendix A in Hinton et al, 2006)
        """
        p_reconstr, _reconstr = self.rbm_layers[-1].sample_v_given_h(_reconstr)
        _reconstr, _ = self.rbm_layers[-1].Gibbs_sampling(_reconstr, mcmc_steps = 20)
        
        for i in range(len(self.rbm_layers)-2,-1,-1):
            print('Layer {}'.format(i))
            _reconstr = self.rbm_layers[i].h_to_v(_reconstr)
            _reconstr = torch.sigmoid(torch.sigmoid(_reconstr))
            # _reconstr = torch.bernoulli(torch.sigmoid(_reconstr))
        #end
        
        visible_reconstructed = _reconstr.clone()
        return visible_reconstructed
    #end
    
    
    def contrastive_divergence_train(self, training_set, epochs, 
                                     learning_rate, weights_decay, momentum,
                                     mcmc_steps):
        """
        Constrastive divergence training 
        
        Call signature: see rbm.RestrictedBoltzmannMachine.contrastive_divergence_train
        it is the same as here.
        """
        
        Xtrain = training_set[0]
        Ytrain = training_set[1]
        
        _input = copy.deepcopy(Xtrain)
        i = 1
        
        for _rbm in self.rbm_layers:
            print('Training RBM {:d}'.format(i))
            print('_'*20)
            
            # vs.plot_images_grid(_input[0], Ytrain[0], title = 'Input of layer {:d}'.format(i))
            
            _rbm.contrastive_divergence_train([_input,Ytrain], epochs,
                           learning_rate, weights_decay, momentum, mcmc_steps)
            
            for j in range(len(_input)):
                # new data -> probabilities of the hidden states
                _hidden, p_hidden = _rbm.sample_h_given_v(_input[j])
                # _input[j]  = _hidden.clone()
                _input[j]  = p_hidden.clone()
            #end
            i += 1
            
            W = self.rbm_layers[0].v_to_h.weight
            for _rbm in self.rbm_layers[1:i]:
                W = torch.mm(W.t(),_rbm.v_to_h.weight.t())
                W = W.t()
            #end
            vs.receptive_fields_visualization(W)
        #end
    #end
    
    
    def reconstruction(self, test_data):
        """
        Reconstruct test samples
        
        See the call signature of rbm.RestrictedBoltzmannMachine.reconstruction
        """
        
        
        Xtest = test_data[0]
        Ytest = test_data[1]
        
        indices = [np.random.randint(0, len(Xtest)) for _ in range(5)]
        Xtest = [Xtest[i] for i in indices]
        Ytest = [Ytest[i] for i in indices]
        
        for i in range(len(indices)):
            reconstructions = self.forward(Xtest[i])
            vs.plot_images_grid(Xtest[i], Ytest[i], title = 'Original samples')
            vs.plot_images_grid(reconstructions, Ytest[i], title = 'Reconstructed samples')
        #end
        
#end
