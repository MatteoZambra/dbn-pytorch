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
        
        _reconstr = visible.clone().cuda()
        
        for i in range(len(self.rbm_layers)):
            # print('Layer {}'.format(i))
            
            # use probabilities to reduce sampling noise
            _reconstr, _ = self.rbm_layers[i].sample_h_given_v(_reconstr)
        #end
        
        _reconstr, _v = self.rbm_layers[-1].sample_v_given_h(_reconstr)
        # _reconstr_prob, _reconstr  = self.rbm_layers[-1].Gibbs_sampling(_v, mcmc_steps = 1)
        
        for i in range(len(self.rbm_layers)-2,-1,-1):
            # print('Layer {}'.format(i))
            
            _reconstr, _ = self.rbm_layers[i].sample_v_given_h(_reconstr)
        #end
        
        return _reconstr
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
        
        _Xtrain = copy.deepcopy(Xtrain)
        
        i = 1
        for _rbm in self.rbm_layers:
            print('Training RBM {:d}'.format(i))
            print('_'*20)
            
            _Xtrain = _rbm.contrastive_divergence_train([_Xtrain, Ytrain], epochs,
                           learning_rate, weights_decay, momentum, mcmc_steps)
            i += 1
            W = self.rbm_layers[0].W
            for _rbm in self.rbm_layers[1:i]:
                W = torch.mm(W.t(),_rbm.W.t())
                W = W.t()
            #end
            vs.receptive_fields_visualization(W)
            
            if i == len(self.rbm_layers)+1:
                
                self.linear_readout(_Xtrain, Ytrain)
            #end
        #end
    #end
    
    
    def linear_readout(self, topend_pattern, labels):
        """
        If the device has been properly trained, it should be possible to
        linearly classify the activity patterns in the top end layer.
        A linear SVM is thought to suffice for such a purpose
        """
        
        # reshape topend_patterns and labels to have a 60000x2000 numpy.ndarray
        # design matrix and 60000x10 numpy.ndarray labels!
        from sklearn import svm
        from sklearn.metrics import accuracy_score
        
        num_batches    = topend_pattern.shape[0]
        batch_size     = topend_pattern.shape[1]
        dataset_length = num_batches * batch_size
        
        topend_patterns = topend_pattern.cpu().numpy().reshape(dataset_length, -1)
        train_patterns  = topend_patterns[:3000,:]
        labels          = labels.cpu().numpy().reshape(dataset_length, -1).ravel()
        train_labels    = labels[:3000]
        test_patterns   = topend_patterns[3000:3200,:]
        test_labels     = labels[3000:3200]
        
        linear_classifier = svm.SVC()
        linear_classifier.fit(train_patterns, train_labels)
        
        predicted_labels = linear_classifier.predict(test_patterns)
        print('Read-out accuracy at the top-end layer: {}'.format(accuracy_score(test_labels, predicted_labels)))
    #end
    
    
    def reconstruction(self, test_data, title):
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
            vs.plot_images_grid(Xtest[i], Ytest[i], title)
            vs.plot_images_grid(reconstructions, Ytest[i], title = 'Reconstructed samples')
        #end
    #end
    
    
    def corrupt_and_recreate(self, Xtest, Ytest):

        side_dim     = int(np.sqrt(Xtest[0][0].shape[0]))
        row          = 3
        rows_to_kill = 5
        
        """
        Corruption.
        The images are vectorized, i.e. 28x28 matrices flattened. 
        Recall the good old row-major order. Assume we want to delete the i to i+k-th 
        rows, setting all the pixels is such rows to 0.0 -- i.e. black. 
        Then all the matrix entries from (i+1)*28 to (i+k)*dim + dim should be selected.
        Considering the flattened matrix, it translated simply to setting to 0.0 the 
        entries of the vector X[(i+1)*dim : (i+k)*dim +dim].
        """
        
        x = Xtest[0] # one batch
        for x in Xtest:
            x[:, (row + 1)*side_dim : (row + rows_to_kill)*side_dim + side_dim] = 0.0
        #end
        
        self.reconstruction([Xtest,Ytest], title = 'Corrupted samples')
    #end
    
    
    def noise_and_denoise(self, Xtest, Ytest, title):
       
        _Xtest = Xtest + torch.normal(0.0, 0.1, Xtest.shape).cuda()
        self.reconstruction([_Xtest, Ytest], 'Noised samples')
    #end
        
#end