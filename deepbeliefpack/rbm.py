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

import torch
from torch import sigmoid, bernoulli, matmul
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
        
        to add:   - Glorot initialization (not necessary. We do not have to 
                    propagate gradients, nor there is the peril of gradients
                    vanishing or exploding)
        
        Input:
            ~ visible_dim (integer)
            ~ hidden_dim  (integer) self-explanatory
            
        Returns:
            nothing
        """
        
        super(RestrictedBoltzmannMachine, self).__init__()
        
        self.W = torch.normal(0.0, 0.01, (hidden_dim, visible_dim)).cuda()
        self.b = torch.zeros(1, hidden_dim).cuda()
        self.a = torch.zeros(1, visible_dim).cuda()
        
        self.input_size  = visible_dim
        self.output_size = hidden_dim
        
        self.velocity_w = torch.zeros_like(self.W).cuda()
        self.velocity_a = torch.zeros_like(self.a).cuda()
        self.velocity_b = torch.zeros_like(self.b).cuda()
    #end
    
    
    def sample_h_given_v(self, visible):
        
        p_h = sigmoid(matmul(visible, self.W.t()) + self.b)
        h   = bernoulli(p_h)
        
        return p_h, h
    #end
    
    
    def sample_v_given_h(self, hidden):
        
        p_v = sigmoid(matmul(hidden, self.W) + self.a)
        v   = bernoulli(p_v)
        
        return p_v, v
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
            ~ visible_pattern  : reconstructed after k full steps (negative data)
            ~ p_visible : probability of the hidden units given the 
                         (reconstructed) visible pattern (negative hidden 
                         probabilities, associated with negative data)
        """
        _v = visible.clone()
        
        for k in range(mcmc_steps):
            
            p_h, h   = self.sample_h_given_v(_v)
            p_v, _v  = self.sample_v_given_h(h)
        #end
        
        p_h, h   = self.sample_h_given_v(_v)
        
        return p_h, _v
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
        cost_values      = []
        
        num_batches = training_set[0].shape[0]
        batch_size  = training_set[0].shape[1]
        _Xtrain = torch.zeros([num_batches, batch_size, self.output_size])
        
        dW = torch.zeros_like(self.W)
        da = torch.zeros_like(self.a)
        db = torch.zeros_like(self.b)
        
        for epoch in range(epochs):
            
            if epoch >= int(len(range(epochs))/2):
                momentum = 0.9
            #end
            
            train_loss = 0.0
            train_set  = training_set[0]
            
            for n in range(num_batches):
                """
                This is a loop over the TRAIN BATCHES. For each batch,
                the model paramters are updated 
                """
                
                histogram_flag = False
                if (epoch == epochs-1 and n == num_batches-1):
                    histogram_flag = True
                #end
                
                """
                Positive phase. Data batches and probabilities of 
                activity of the hidden variables associated with data
                """
                pos_v  = train_set[n,:,:]
                pos_ph = sigmoid(matmul(pos_v, self.W.t()) + self.b)
                pos_dW = matmul(pos_v.t(), pos_ph).t()
                pos_da = torch.sum(pos_v, dim = 0)
                pos_db = torch.sum(pos_ph, dim = 0)
                # pos_h  = bernoulli(pos_ph)
                
                
                
                """
                Negative phase. Fantasy particles are generated to be 
                used as fake data. Block Gibbs sampling with either 1
                or more MCMC steps
                """
                # neg_v  = sigmoid(matmul(pos_h, self.W) + self.a)
                # neg_ph = sigmoid(matmul(neg_v, self.W.t()) + self.b)
                neg_ph, neg_v = self.Gibbs_sampling(pos_v, mcmc_steps = 1)
                neg_dW = matmul(neg_v.t(), neg_ph).t()
                neg_da = torch.sum(neg_v, dim = 0)
                neg_db = torch.sum(neg_ph, dim = 0)
                
                dW = momentum * dW + learning_rate * ((pos_dW - neg_dW) / batch_size - weights_decay * self.W)
                da = momentum * da + learning_rate * (pos_da - neg_da) / batch_size
                db = momentum * db + learning_rate * (pos_db - neg_db) / batch_size
                
                self.W = self.W + dW
                self.a = self.a + da
                self.b = self.b + db
                
                if histogram_flag:
                    vs.parameters_histograms(self.W, dW,self.a, da, self.b, db)
                #end
                
                loss = criterion(pos_v, neg_v)
                train_loss += loss.item()
                
                if epoch == epochs-1:
                    _Xtrain[n,:,:] = sigmoid(matmul(pos_v, self.W.t()) + self.b)
                #end
            #end
            
            cost_values.append(train_loss / batch_size)
            print('Epoch {:02d} Training loss = {:.6f}'.format(epoch+1, train_loss / batch_size))
        #end
        
        vs.cost_profile_plot(cost_values)
        return _Xtrain.cuda()
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