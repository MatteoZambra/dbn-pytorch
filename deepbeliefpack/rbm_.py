

import images_utils as iu
import numpy as np
import torch
from datetime import datetime


class RBM:
    
    def __init__(self, visible_dim, hidden_dim):
        """
        Constructor.
        Connection strenghts are initialized as small random values, as
        customary. Biases are initialized to zero.
        With consistent dimensions, the velocity tensors are initialized 
        to zeros as well. These will be needed for momentum addition in
        training stage
        
        Input:
            ~ visible_dim: visible layer dimensions, visible units
            ~ hidden_dim:  hidden units
        
        Returns: 
            nothing
        """
        
        self.W = torch.normal(0.0, 0.01, (hidden_dim, visible_dim))
        self.a = torch.randn(1, visible_dim)
        self.b = torch.randn(1, hidden_dim)
        
        self.velocity_w = torch.zeros_like(self.W)
        self.velocity_a = torch.zeros_like(self.a)
        self.velocity_b = torch.zeros_like(self.b)
    #end
    
    def sample_hidden_given_visible(self, visible):
        """
        Handiness of RBMs, or better posed, the ease to train them with 
        respect to plain Boltzmann machines, stems from the 
        intra-independentness of both visible and hidden units, hence 
        yielding the ease of probability density function factorization.
        Sampling is simple, it sufficies to sample an hidden pattern
        according to the distribution
            
            p(h_i | v) = sigmoid(b + Wv)
        
        Input:
            ~ visible: a data batch. From this data, one can compute the 
                       hidden patterns, according to the above expression
                       of the probability distribution
                       
        Returns:
            ~ p_hidden_given_visible : p(h_i | v). Vector of probabilities
            ~ hidden_pattern:          activity pattern of the hidden units
            
        Both are subsequently needed
        """
        
        activity = torch.matmul(visible, self.W.t()) + self.b
        p_hidden_given_visible = torch.sigmoid(activity)
        hidden_pattern = torch.bernoulli(p_hidden_given_visible)
        return p_hidden_given_visible, hidden_pattern
    #end
    
    def sample_visible_given_hidden(self, hidden):
        """
        Likewise, but the given pattern is that of the hidden units and the
        sampling involves visible units activities. It is a reconstruction
        of the data samples
        """
        
        activity = torch.matmul(hidden, self.W) + self.a
        p_visible_given_hidden = torch.sigmoid(activity)
        visible_pattern = torch.bernoulli(p_visible_given_hidden)
        return p_visible_given_hidden, visible_pattern
    #end
    
    
    def params_update(self, visible_0, visible_k, p_hidden_0, p_hidden_k,
                      learning_rate, weight_decay, momentum, histogram):
        """
        Parameters are updated accounting for the momentum and the weights 
        decay caveats. In formulae
        
            w(t+1) = w(t) + learing_rate * d/dw (ln L(w(t))) -
                            weight_decay * w(t) + 
                            momentum * Delta_w(t-1)
                        
        The velocity Delta_w(t-1) is retained and saved as a filed of the RBM
        class. Subsequently, all the right hand side of the current parameters
        update is stored in the velocities saved, inasmuch it all represents 
        that quantity.
        See Fischer and Igel (2014) for an extensive account:
        > https://www.sciencedirect.com/science/article/pii/S0031320313002495
        
        Input:
            ~ visible_0, visible_k,
              hidden_0, hidden_k:   visible and hidden activities at steps
                                    0 and k of the Gibbs sampling. 
            ~ learning_rate,
              weight_decay,
              momentum:             as above
            ~ histogram:            boolean, whether plotting the histrograms
                                    or not
        
        Returns:
            nothing
        """
        
        product_1 = torch.matmul(visible_0.t(), p_hidden_0)
        product_2 = torch.matmul(visible_k.t(), p_hidden_k)
        
        W_update = (product_1 - product_2).t()
        a_update = torch.sum(visible_0 - visible_k, 0)
        b_update = torch.sum(p_hidden_0 - p_hidden_k, 0)
        
        if (histogram):
            iu.plot_params_histogram_(self.W, learning_rate * W_update,
                                      self.a, learning_rate * a_update,
                                      self.b, learning_rate * b_update)
        #end
        
        W_add = learning_rate * W_update - weight_decay * self.W + momentum * self.velocity_w
        a_add = learning_rate * a_update - weight_decay * self.a + momentum * self.velocity_a
        b_add = learning_rate * b_update - weight_decay * self.b + momentum * self.velocity_b
        
        self.W += W_add
        self.a += a_add
        self.b += b_add
        
        self.velocity_w = torch.clone(W_add)
        self.velocity_a = torch.clone(a_add)
        self.velocity_b = torch.clone(b_add)
    #end
    
    
    def Gibbs_sampling(self, visible, steps):
        """
        Gibbs sampling: from the provided data samples generates the
        reconstruction of activity patterns of the visible units
        
        Input: 
            ~ visible: visible units activities
            ~ steps:   mcmc steps to perform
        
        Returns:
            ~ visible: reeconstruction of the visible units
        
        Hinton (2010) argues that the last sampling of the hidden patterns
        should use the probabilities instead of the binarized probabilities
        to avoid sampling noise
        """
        
        for k in range(steps):
            
#            
            if (k == steps-1):
                hidden, _ = self.sample_hidden_given_visible(visible)
                _, visible = self.sample_visible_given_hidden(hidden)
            else:
                _, hidden = self.sample_hidden_given_visible(visible)
                _, visible = self.sample_visible_given_hidden(hidden)
        #end
        
        return visible
    #end
    
    def train(self, train_set, epochs, 
              learning_rate, weight_decay, momentum, 
              mcmc_steps = 20):
        """
        training batches are extracted from the data_iterator.
        
        Training is performed on a zip of two lists containing 
        data batches and labels. In the following data list creation, 
        batches are appended after being flattened, that is, in a 
        format compatible with the RBM architecture
        
        Input:
            ~ trian_set:    torch.utils.data.DataLoader type. It is possible
                            iterate over the iter(train_set) to loop
                            over data batches and labels
            ~ epochs:       number of epochs
            ~ learing_rate: learning rate
            ~ momentum:     fraction of ``velocity'' to add to the current update
            ~ weight_decay: regularization quantity. How much of the current
                            weights values to subtract from the update
            ~ mcmc_steps:   number of Gibbs sampling steps being performed to
                            sample the visible pattern given the hidden units
                            pattern (given a data sample)
                            
        Returns:
            nothing
            
        Note that Contrastive Divergence training is performed, hence the 
        activities of visible and hidden units are accounted for at the 
        initial step, that is, as absorbed, and at a k-th arbitrary step of
        the Gibbs sampling process. This yields reconstructed visible patterns
        that approach the thermal equilibrium
        """
        
        
        """
        Un-comment the following two lines to perform training on a 
        smaller subset of data samples. 
        It speeds up training, but worsen the performance
        """
#        idx = [np.random.randint(0, len(train_set)) for _ in range(5)] # cioè scegliamo 100 minibatches
#        train_set = [train_set[i] for i in idx]
        
        print("Training start: " + str(datetime.now()))
        
        criterion = torch.nn.MSELoss()
        plot_epoch_count = 1
        cost = []
        
        for epoch in range(epochs):
            
            train_loss = 0.0
            batch_item_count = 1
            
            for train_batch,labels_batch in zip(Xtrain,Ytrain):
                
                # histogram plot per 10 epochs
                # plots are produced once the last batch
                # has been processed
                histogram_flag = False
                if (plot_epoch_count % 10 == 0 and 
                    batch_item_count == len(data)): 
                    histogram_flag = True
                #end
                
                train_batch = iu.binarize_digits(train_batch, factor = 3.5)
                visible_0 = train_batch
                visible_k = visible_0
                p_hidden_0, _ = self.sample_hidden_given_visible(visible_0)
                
                visible_k = self.Gibbs_sampling(visible_k, mcmc_steps)
                p_hidden_k, _ = self.sample_hidden_given_visible(visible_k)
                
                self.params_update(visible_0, visible_k, p_hidden_0, p_hidden_k,
                                   learning_rate, weight_decay, momentum, 
                                   histogram_flag)
                
                loss = criterion(visible_0, visible_k)
                
                train_loss += loss.item()
                batch_item_count += 1
            #end
            
            if (plot_epoch_count % 10 == 0):
                iu.receptive_fields_plot(self.W, self.a, self.b)
            #end
            
            plot_epoch_count += 1
            s = len(train_batch)
            cost.append(train_loss/s)
            print("Epoch {:d} \t Training loss = {:.6f}\n---".format(epoch+1, train_loss/s))
        #end
        
        print("Training end " + str(datetime.now()))
        iu.cost_profile_plot(cost)
    #end            
                       
                
    def generate_samples(self, images, labels):
        """
        Absorb test samples and test the models performance upon those
        A Gibbs sampling is performed, taking as initial values held-out
        data instances
        
        Input:
            ~ images: torch.Tensor data samples
            ~ labels: integer associated to categories
        
        Returns:
            ~ samples: reconstructed images. Once returned, they are plotted
        """
        
#        _,hidden_pattern = self.sample_hidden_given_visible(images)
#        _,samples = self.sample_visible_given_hidden(hidden_pattern)
        samples = self.Gibbs_sampling(images, steps = 100)
        iu.images_plot(images.view(-1,28,28),labels)
        iu.images_plot(samples.view(-1, 28,28),labels)
        
        criterion = torch.nn.MSELoss()
        loss = criterion(images,samples)
        s = images.shape[0]
        print("Test loss = {:.6f}".format(loss.item()/s))
        
        return samples
    #end
    
#end






















