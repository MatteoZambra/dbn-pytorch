import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


#np.random.seed(618)

def plot_single_image(image):
    """
    A single digit is displayed.
    Almost useless
    
    Input:
        ~ image (torch.Tensor) : data
        
    Returns:
        ~ nothing
    """
    image = image.cpu()
    
    assert type(image) is torch.Tensor, 'Image to plot is not torch.Tensor'
    image_size = int(np.sqrt(image.shape[0]))
    image = image.view(image_size, image_size)
    
    fig = plt.imshow(image, cmap = 'gray')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.show()
    plt.close('all')
#end


def plot_images_grid(images, labels, title):
    """
    Plot a grid of 2x5 digits
    
    Input:
        ~ images (torch.Tensor) : images data
        ~ labels (torch.Tensor) : labels
        ~ title (string) : this function can be used in many scopes, a title may
                           render clearer 
        
    Returns:
        ~ nothing
    """
    images = images.cpu()
    labels = labels.cpu()
    
    assert type(images[0]) is torch.Tensor, 'Image to plot is not torch.Tensor'
    image_size = int(np.sqrt(images[0].shape[0]))
    
    fig = plt.figure(figsize=(10,4))
    for idx in range(10):
        ax = fig.add_subplot(2,10/2,idx+1, xticks=[], yticks=[])
        ax.imshow(images[idx].view(image_size, image_size), cmap = 'gray')
        label = labels[idx].item()
        ax.set_title(label)
    #end
    fig.suptitle(title, fontsize = 14)
    plt.show()
    plt.close('all')
#end


def parameters_histograms(w, dw, a, da, b, db):
    """
    As pointed out in Hinton (2010), a good sanity check to monitor
    the training process is to inspect the parameters -and variations-
    histograms. 
    
    Input:
        ~ X, dX (torch.Tensor) : quantities to plot the histograms of
        X = weights, visible bias and hidden bias
    
    Returns:
        ~ nothing
    """
    w = w.cpu()
    dw = dw.cpu()
    a = a.cpu()
    da = da.cpu()
    b = b.cpu()
    db = db.cpu()
    
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(231)
    ax.hist(w.reshape(1, w.shape[0] * w.shape[1]))
    ax.set_title('Weights', fontsize = 11)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax = fig.add_subplot(232)
    ax.hist(dw.reshape(1, dw.shape[0] * dw.shape[1]))
    ax.set_title('Weights variations', fontsize = 11)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax = fig.add_subplot(233)
    ax.hist(a)
    ax.set_title('Visible bias', fontsize = 11)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax = fig.add_subplot(234)
    ax.hist(da)
    ax.set_title('Visible bias variations', fontsize = 11)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax = fig.add_subplot(235)
    ax.hist(b)
    ax.set_title('Hidden bias', fontsize = 11)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax = fig.add_subplot(236)
    ax.hist(db)
    ax.set_title('Hidden bias variations', fontsize = 11)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.subplots_adjust(hspace=0.25)
    plt.show()
    plt.close('all')
#end


def receptive_fields_visualization(W):
    """
    Plot receptive fields, that is the combination of weights matrices.
    This is done to visualize what are the features that are learned by the 
    device. Some resemblances with the actual digits could be recognisable.
    Panels plotted are arbitrary, neurons from which these come are chosen randomly
    
    Input:
        ~ W (torch.Tensor) : weights matrix
        
    Returns:
        ~ nothing
    """
    W = W.cpu()
    
    hidden_dim = int(np.sqrt(W.shape[1]))
    side_dim = 10
    indices = [np.random.randint(0,W.shape[0]) for _ in range(side_dim**2)]
    
    fig = plt.figure(figsize=(10,10))
    for i in range(len(indices)):
        ax = fig.add_subplot(side_dim, side_dim, i+1, xticks = [], yticks = [])
        ax.imshow(W[i,:].view(hidden_dim, hidden_dim),cmap = 'gray')
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
    #end
    
    plt.show()
    plt.close('all')    
#end


def recursive_receptive_fields(dbn):
    """
    Once the DBN has been trained, it is possible to compose the receptive fields
    as combinations of weight matrices. See
    ~ Zorzi, Testolin and Stoianov (2013) Modeling language and cognition with
                                          deep unsupervised learning: a tutorial
                                          overview.
                    
    Input:
        ~ dbn (dbn.DeepBeliefNet) : the trained DBN model. Weights are available
                                    as the modules parameters, where each module
                                    is a rbm.RestrictedBoltzmannMachine module
    
    Returns:
        ~ nothing
    """
    
    W = dbn.rbm_layers[0].v_to_h.weight
    receptive_fields_visualization(W)
    for _rbm in dbn.rbm_layers[1:]:
        W = torch.mm(W.t(), _rbm.v_to_h.weight.t())
        W = W.t()
        receptive_fields_visualization(W)
    #end
#end


def cost_profile_plot(cost_values):
    """
    Plot the MSE profile, values of which are recorded during 
    training.
    
    Input:
        ~ cost_values (list of floats) : self-explanatory
        
    Returns:
        ~ nothing
    """
    
    ax = plt.figure(figsize = (7.5,4.5)).gca()
    cost_values = np.array(cost_values)
    span = np.arange(1,len(cost_values)+1)
    ax.plot(span,cost_values, color = 'k', alpha = 0.7)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cost (MSE) value')
    plt.show()
    plt.close('all')
#end