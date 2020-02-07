import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms



class LoadDataset:
    """
    Class LoadDataset. It serves the purpose of dowloading, preprocessing and
    fetching the data samples, divided in training set and testing set.
    
    Currently, this class is thought only for the MNIST data set. Enhancement
    in order to fetch different torchvision.datasets items is easy.
    However, a greater flexibility, to include more classes of data, even custom
    ones could require more effort and design.
    """

    def __init__(self, batch_size, transforms):
        """
        Constructor.
        Note that the transform used is that which transforms to tensors the 
        images. Samples are normalized yet.
        
        Input:
            ~ batch_size (integer) : batch size, the number of samples to train
                                     the device with, iteratively
            ~ transforms (list of torchvision.transforms) : how to pre-process
                                                            the data samples
        
        Returns:
            ~ nothing
        """
        
        self.batch_size = batch_size
        self.transforms = transforms
    #end        
        
    
    def yield_data(self, binarize, factor):
        """
        The strategy is the following. 
        Data set is provided by the torchvision.datasets module. MNIST is 
        there available. With the torch.utils.data.DataLoader one can obtain
        an iterable object out of the MNIST object. 
        Training set and testing set, respectively Xtrain,Ytrain and Xtest,Ytest
        are obtained by fetching each item from the iterators on the training 
        and testing sets. The whole data set is then available as lists of data
        batches, which are subsequently used in this compact format.
        
        Input:
            ~ binarize (boolean) : if True, the data samples are transformed in
                                   {0,1} numerical matrices
            ~ factor (float) : if binarize == True, then the numerical values of
                               the features are binarized according to this value
                               
        Returns:
            ~ Xtrain and Xtest (list of torch.Tensor) : each tensor has shape
                                                        [batch_size, num_features]
                                                        the lists lengths are different
            ~ Ytrain and Ytest (list of torch.Tensor) : these items have shape
                                                        [batch_size, 1]. Labels 
                                                        associated with data samples
        """
        
        transfs = transforms.Compose(self.transforms)
        
        train_data = MNIST(r'data/', download = True, train = True,  transform = transfs)
        test_data  = MNIST(r'data/', download = True, train = False, transform = transfs)
        
        train_load = DataLoader(train_data, batch_size = self.batch_size, shuffle = False)
        test_load  = DataLoader(test_data,  batch_size = self.batch_size, shuffle = False)
        
        data_iterator  = iter(train_load)
        Xtrain, Ytrain = self.iter_to_list(list(data_iterator))
        
        data_iterator  = iter(test_load)
        Xtest, Ytest   = self.iter_to_list(list(data_iterator))
        
        if binarize:
            Xtrain = self.binarize_digits(Xtrain, factor)
            Xtest  = self.binarize_digits(Xtest,  factor)
        #endif
        
        return Xtrain,Ytrain, Xtest,Ytest
    #end
    
    
    def yield_tensor_data(self):
        
        transfs = transforms.Compose(self.transforms)
        
        train_data = MNIST(r'data/', download = True, train = True,  transform = transfs)
        test_data  = MNIST(r'data/', download = True, train = False, transform = transfs)
        
        train_load = DataLoader(train_data, batch_size = self.batch_size, shuffle = False)
        test_load  = DataLoader(test_data,  batch_size = self.batch_size, shuffle = False)
        
        train_iterator = iter(train_load)
        test_iterator  = iter(test_load)
        
        Xtrain = torch.Tensor()
        Xtest  = torch.Tensor()
        Ytrain = torch.LongTensor()
        Ytest  = torch.LongTensor()
        
        for data, labels in train_iterator:
            Xtrain = torch.cat([Xtrain, data], 0)
            Ytrain = torch.cat([Ytrain, labels], 0)
        #end
        
        Xtrain = Xtrain.view(-1,28*28)  # view(-1,28,28)
        size_dataset = int(Xtrain.shape[0] / self.batch_size)
        Xtrain = Xtrain.view(size_dataset, self.batch_size, 28*28)
        Ytrain = Ytrain.view(size_dataset, self.batch_size, 1)
        
        for data, labels in test_iterator:
            Xtest  = torch.cat([Xtest, data], 0)
            Ytest  = torch.cat([Ytest, labels], 0)
        #end
        
        Xtest = Xtest.view(-1,28*28)  # view(-1,28,28)
        size_dataset = int(Xtest.shape[0] / self.batch_size)
        Xtest = Xtest.view(size_dataset, self.batch_size, 28*28)
        Ytest = Ytest.view(size_dataset, self.batch_size, 1)
        
        Xtrain = Xtrain.cuda()
        Xtest  = Xtest.cuda()
        
        return Xtrain, Xtest, Ytrain, Ytest
    #end
    
    @staticmethod
    def iter_to_list(data_set):
        """
        Transforms iterators to lists.
        Each list contains the data samples, labels.
        
        Input:
            ~ data_set (torch.utils.data.dataloader._SingleProcessDataLoaderIter) : 
                iterator over the torch.utils.data.dataloader.DataLoader object.
        
        Returns:
            ~ data (list of torch.Tensor) : list of data batches
            ~ labels (list of torch.Tensor) : list of labels associated with the
                                              samples in the data batches in data
        """
        
        data   = []
        labels = []
        
        for _data,_labels in data_set:
            _data = _data.view(-1, _data.shape[2] * _data.shape[3])
            data.append(_data)
            labels.append(_labels)
        #enddo
        
        return data,labels
    #end
    
    @staticmethod
    def binarize_digits(data_set, factor):
        """
        Images binarization. 
        The threshold value for binarization is set to 1/factor, and the pixels
        in the image which exceed that value are set to 1, the others are shut
        to 0. The larger the factor, the less pixels survive.
        
        Input:
            ~ data_set (list of torch.Tensor) : could be Xtrain or Xtest
            ~ factor (float) : denominator of thethreshold value computation
            
        Returns:
            ~ data_set (list of torch.Tensor) : binarized images of the train/test
                                                data samples sets
        """
        
        threshold = 1.0 / factor
        
        for x in data_set:
            for sample in x:
                sample[sample <= threshold] = 0.
                sample[sample >  threshold] = 1.
            #end
        #end
        return data_set
    #end
#end