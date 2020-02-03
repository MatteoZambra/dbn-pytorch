
"""
Class restriced Boltzmann machine. 
Provides the nn.Module subclass containing all what needed to 
train such a device. 

At the present stage, the RBM is thought to be the building block of
a Deep Belief Network. The training procedure for this latter relies
on the Contrastive Divergence, which is also used in this scope indeed.

For a deep view on the subject see the following references

~ Hinton (2002) Training Products of Experts by Minimizing Contrastive Divergence
~ Hinton Osindero and Teh (2006) A Fast Learning Algorithm for Deep Belief Nets
~ Hinton (2010) A Practical Guide to Training Restricted Boltzmann Machines
~ Fischer and Igel (2014) Training Restricted Boltzmann Machines: An Introduction

"""
import random
import copy

import torch
import torch.nn as nn

import numpy as np
import visual as vs


torch.manual_seed(618)

class RestrictedBoltzmannMachine(nn.Module):
    
    def __init__(self, visible_dim, hidden_dim):
        """
        Constructor
        * Motivations
        ** forward and backward passes: It is pivotal in RBMs to perform 
        forth and back transformations of a given data instance, expecially 
        in the Gibbs sampling stage during training. 
        Hence it is thought useful to endow the RBM class with the linear 
        transformations
            _h = v*W' + b
            _v = h*W  + a
        
        Here the ' denotes the transposition of the weights matrix W and a, b 
        denote the visible and hidden layers biases respectively.
        Pytorch allows to define such transformations, that are subsequently 
        post-processed to yield probabilities and visible/hidden activities,
        by means of the torch.sigmoid(torch.Tensor) method. In turn, activity 
        patterns are extrapolated by the torch.bernoulli(torch.Tensor) method: 
        The value activity[i] is set to 1 with probability probabilities[i]. 
        
        to add:   - Glorot initialization
        
        Input:
            ~ visible_dim (integer)
            ~ hidden_dim  (integer) self-explanatory
            
        Returns:
            nothing
        """
        
        super(RestrictedBoltzmannMachine, self).__init__()
        
        _W = torch.normal(0.0, 0.01, (hidden_dim, visible_dim))
        _b = torch.zeros(1, hidden_dim)
        _a = torch.zeros(1, visible_dim)
        
        self.v_to_h = nn.Linear(visible_dim, hidden_dim)
        self.h_to_v = nn.Linear(hidden_dim, visible_dim)
        
        self.h_to_v.weight = nn.Parameter(_W.t(), requires_grad = False)
        self.h_to_v.bias   = nn.Parameter(_a, requires_grad = False)
        self.v_to_h.weight = nn.Parameter(_W, requires_grad = False)
        self.v_to_h.bias   = nn.Parameter(_b, requires_grad = False)
        
        self.velocity_w = torch.zeros_like(self.v_to_h.weight)
        self.velocity_a = torch.zeros_like(self.h_to_v.bias)
        self.velocity_b = torch.zeros_like(self.v_to_h.bias)
    #end
    
    
    def forward(self, visible_pattern):
        """
        As usual for Pytorch modules, the forward method isntructs the program
        what the network is supposed to do, from input to output, disposing of
        the layers declared in the constructor.
        
        In the scope of restricted Boltzmann machines, this dynamics is thought
        to be accomplished by the reconstruction of a given input. This is
        done once the model parameters have been optimized.
        
        Input:
            ~ visible_pattern (torch.Tensor) : given data
        
        Returns:
            ~ recostruction (torch.Tensor) : reconstruction of the input
        """
        
        hidden_pattern = torch.bernoulli(torch.sigmoid(self.v_to_h(visible_pattern)))
        reconstruction = torch.bernoulli(torch.sigmoid(self.h_to_v(hidden_pattern)))
        return reconstruction
    #end
    
    
    def sample_h_given_v(self, visible_pattern):
        """
        Samples h ~ p(h|v) 
        
        Input:
            ~ visible_pattern (torch.Tensor) : data instance
            
        Returns:
            ~ p_hidden_given_visible (torch.Tensor) : probabilities associated with hidden units activities
            ~ hidden_pattern (torch.Tensor) : hidden activity
        """
        
        p_hidden_given_visible = torch.sigmoid(self.v_to_h(visible_pattern))
        hidden_pattern = torch.bernoulli(p_hidden_given_visible)
        
        return p_hidden_given_visible, hidden_pattern
    #end
    
    
    def sample_v_given_h(self, hidden_pattern):
        """
        Samples v ~ p(v|h)
        
        Input:
            ~ hidden_pattern (torch.Tensor) : hidden units activity
        
        Returns:
            ~ p_visible_given_hidden (torch.Tensor) : probabilities associated with visible units activities
            ~ visible_pattern (torch.Tensor) : visible activities
        """
        
        p_visible_given_hidden = torch.sigmoid(self.h_to_v(hidden_pattern))
        visible_pattern = torch.bernoulli(p_visible_given_hidden)
        
        return p_visible_given_hidden, visible_pattern
    #end
    
    
    def Gibbs_sampling(self, visible, mcmc_steps):
        """
        Repeatedly sample from p(h|v) and p(v|h) starting from a given
        datum v. This is done for the purpose of getting samples from
        the model equilibrium distribution. Such samples are then unsed 
        to compute the correlations between visible and hidden activities
        that are plugged in the update delta rule:
            Delta w_ij = lr * ( < v_i * h_j >_data - < v_i * h_j >_model )
            
        Input:
            ~ visible --- as usual
            ~ mcmc_steps (integer) : how many Gibbs sampling steps to perform
                                     For most applications even one full step
                                     has been verified to suffice
            
        Returns:
            ~ visible_pattern  : reconstructed after k full steps
            ~ p_visible : probability of the hidden units given the 
                         (reconstructed) visible pattern
        """
        
        for k in range(mcmc_steps):
            
            p_hidden, hidden   = self.sample_h_given_v(visible)
            p_visible, visible = self.sample_v_given_h(hidden)
        #end
        
        return p_visible, visible
    #end


    """
    Note that the above methods have no effects on the model parameters yet.
    Rather are just to sample hidden/visible activities and Gibbs sample.
    
    In the following, training methods are provided
    """
    
    def contrastive_divergence_train(self, training_set, epochs,
                                     learning_rate, weights_decay, momentum,
                                     mcmc_steps):
        """
        Contrastive Divergence training. 
        Workflow is as follows:
        For each epoch 1, ..., epochs:
            1. start with a data batch X
            2. data instances are reconstructed after k Gibbs sampling steps
            3. correlations <v_i * h_j> are computed using both the original
               data vector v_0 and the associated hidden activity h_0 and the 
               reconstructed data vector v_k and the associated hidden pattern
               h_k, alongside with the relative probabilities pv and ph
               for the steps 0 and k. 
            4. weights and biases are corrected according to the delta rule
                Delta w_ij = lr * ( < v_i * h_j >_data - < v_i * h_j >_model )
            5. repeat
        
        Mean Squared Error is used as reconstruction error. 
        Note that this metric could be misleading, since it does not descend
        directly from the objective function implemented (Contrastive Divergence).
        Nevertheless it could be used, but should not blindly trusted, see the 
        RBMs guide by Hinton (2010). It is therein pointed out that one expects 
        the MSE to drop abruptly in the early stages of training and it 
        subsequently stabilizes to a carrying value.
        However, a more reliable clue is offered by the weights histograms. See
        the paramters_histogram function description in the visual.py module.
        
        Input:
            ~ training_set (list of torch.Tensor) : contains the training data 
                                                    and the respective labels
            ~ epochs (integer) : number of epochs
            ~ learning_rate (float) 
            ~ weights_decay (float)
            ~ momentum (float) : hyper-parameters
            ~ mcmc_steps (integer) : number of Gibbs sampling steps to perform
        
        Returns:
            ~ nothing
            
        Uncomment the following two lines to perform training on a subset, 
        the numerosity of which should be specified in the main file, of data
        batches. 
        Note that the training_set given as input is a list, the first entry 
        of which is a list itself, every element of which is in turn a torch.Tensor
        of shape (batch_size, number_of_features). The underlying logic is to
        provide data samples as rows of each of such tensors.
        If training is performed on few batches among all those available in
        training_set[0], then training stage is faster, but performance is 
        heavily affected. It more suited for debug purposes
        
        
        TO IMPROVE: training should be flexible enough to allow the user to
        choose other algorithms. Necessary to write two override methods that 
        respond differently whether the training is chosen to be CD reliant or not.
        """
        
        criterion        = nn.MSELoss()
        count_plot_epoch = 1
        cost_values      = []
        
        for epoch in range(epochs):
            
            train_loss       = 0.0
            count_batch_item = 1
            
            Xtrain = copy.deepcopy(training_set[0])
            Ytrain = copy.deepcopy(training_set[1])
            
            # shuffle training set of data and labels
            train_set_tmp = list(zip(Xtrain, Ytrain))
            random.shuffle(train_set_tmp)
            Xtrain, Ytrain = zip(*train_set_tmp)
            
            # debug purpose only
#            number_subbatches = 5
#            indices = [np.random.randint(0, len(Xtrain)) for _ in range(number_subbatches)]
#            Xtrain = [Xtrain[i] for i in indices]
#            Ytrain = [Ytrain[i] for i in indices]
            
            for train_batch,labels in zip(Xtrain, Ytrain):
                """
                This is a loop over the TRAIN BATCHES. For each batch,
                the model paramters are updated 
                """
                
                histogram_flag = False
                if (count_plot_epoch % epochs == 0 and count_batch_item == len(training_set[0])):
                    histogram_flag = True
                #end
                
                visible_0 = train_batch.clone()
                visible_k = visible_0.clone()
                
                p_hidden_0, hidden_0   = self.sample_h_given_v(visible_0)
                visible_k, _ = self.Gibbs_sampling(visible_0, mcmc_steps = 2)
                p_hidden_k, _ = self.sample_h_given_v(visible_k)
                
                self.parameters_update(visible_0, visible_k, p_hidden_0, p_hidden_k,
                                       learning_rate, weights_decay, momentum,
                                       histogram_flag)
                
                loss = criterion(visible_0, visible_k)
                train_loss += loss.item()
                count_batch_item += 1
            #end
            
#            if (count_plot_epoch % epochs == 0):
#                vss.receptive_fields_visualization(self.v_to_h.weight.data)
#            #end
            
            count_plot_epoch += 1
            cost_values.append(train_loss / len(train_batch))
            print('Epoch {:02d} Training loss = {:.6f}'.format(epoch+1, train_loss / len(train_batch)))
#            print('-'*20)
        #end
        
        vs.cost_profile_plot(cost_values)
    #end   
    
        
    def parameters_update(self, visible_0, visible_k, p_hidden_0, p_hidden_k,
                          learning_rate, weights_decay, momentum, histogram_flag):
        """
        Updating the model paramters according to the delta rules
        
        Delta w = lr * ( < v.h >_0 - < v.h >_k )
        Delta a = lr * ( < v >_0 - < v >_k )
        Delta b = lr * ( < h >_0 - < h >_k )
        
        Input:
            ~ visible_0,_k (torch.Tensor) : visible patterns, data and 
                                            reconstructions respectively
            ~ p_hidden_0,_k (torch.Tensor) : probabilities associated with
                                             hidden activities, from steps 0 and k
            ~ learning_rate, weights_decay, momentum (floats) : hyper-parameters
            ~ histogram_flag (boolean) : instructs the program about whether 
                                         to plot the paramters histograms or not
                                         
        Returns:
            ~ nothing
        """
        batch_size = visible_0.shape[0]
        
        p_data_correlation  = torch.matmul(visible_0.t(), p_hidden_0)
        p_model_correlation = torch.matmul(visible_k.t(), p_hidden_k)
        
        W_update = (p_data_correlation - p_model_correlation).t()
        a_update = torch.sum(visible_0 - visible_k, dim = 0)
        b_update = torch.sum(p_hidden_0 - p_hidden_k, dim = 0)
        
        if (histogram_flag):
            vs.parameters_histograms(self.v_to_h.weight.data, learning_rate * W_update / batch_size,
                                     self.v_to_h.bias.data,   learning_rate * a_update / batch_size,
                                     self.h_to_v.bias.data,   learning_rate * b_update / batch_size)
        #end
        
        W_add = momentum * self.velocity_w + learning_rate * (W_update / batch_size - weights_decay * self.v_to_h.weight)
        a_add = momentum * self.velocity_a + learning_rate * a_update / batch_size
        b_add = momentum * self.velocity_b + learning_rate * b_update / batch_size
        
        self.v_to_h.weight.data += W_add
        self.h_to_v.bias.data   += a_add
        self.v_to_h.bias.data   += b_add
        self.h_to_v.weight.data  = self.v_to_h.weight.t()
        
        self.velocity_w = torch.clone(W_add)
        self.velocity_a = torch.clone(a_add)
        self.velocity_b = torch.clone(b_add)
    #end
    
    """
    Once training is done, to test the RBM, one feeds it some test samples.
    Note that sampling from p(h) is infeasible, hence one can  propagate the 
    signal v from the visible layer to the deepest layer and then propagating
    this signal back. The reconstruction should be sufficiently resembling to
    the original data vector
    """
    
    def reconstruction(self, test_data):
        """
        Reconstruction of the visible layer activities, the test data
        
        Input:
            ~ test_data (list of torch.Tensor) : test_data[0] is the list of 
                                                 test data batches. test_data[1]
                                                 are the respective labels
                                                 
        Returns:
            ~ nothing
        """
        
        Xtest = test_data[0]
        Ytest = test_data[1]
        
        indices = [np.random.randint(0, len(Xtest)) for _ in range(3)]
        Xtest = [Xtest[i] for i in indices]
        Ytest = [Ytest[i] for i in indices]
        
        for i in range(len(indices)):
            reconstructions = self.forward(Xtest[i])
            vs.plot_images_grid(Xtest[i], Ytest[i], title = 'Original samples')
            vs.plot_images_grid(reconstructions, Ytest[i], title = 'Reconstructed samples')
        #end
    #end

#endclass

























