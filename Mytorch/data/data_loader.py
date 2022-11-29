import random
import sys
import numpy as np

from scipy.io import loadmat

from ..tensor import*


class DataModule():
    def __init__(self):
        pass

    def get_dataloader(self,train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return  self.get_dataloader(train=False)

class MnistData(DataModule):
    def __init__(self,path,batch_size = 100,reshape=False):
    # 
        self.batch_size = batch_size
        m = loadmat(path)
        trainData, train_labels = m['trainData'], m['trainLabels']
        testData, test_labels = m['testData'], m['testLabels']

        self.X_train = trainData.reshape(-1, 10000).transpose(1, 0)
        self.X_test = testData.reshape(-1, 2000).transpose(1, 0)
       
        self.Y_train = train_labels.transpose(1, 0)
        self.Y_test = test_labels.transpose(1, 0) 
        self.num_train = self.X_train.shape[0]
        self.num_val = self.X_test.shape[0]

        if reshape:
            self.X_train = trainData.reshape((self.num_train, 1,28,28))
            self.X_test = testData.reshape((self.num_val, 1,28,28))

    # same as SVHNloader.get_dataloader
    def get_dataloader(self,train=True):
        if train:
            indices = list(range(self.num_train))
            random.shuffle((indices))
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i: min(i + self.batch_size,self.num_train)]
                yield Tensor(self.X_train[batch_indices]),  Tensor(self.Y_train[batch_indices])
        else:
            indices = list(range(self.num_val))
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i: min(i + self.batch_size,self.num_val)]
                yield  Tensor(self.X_test[batch_indices]),  Tensor(self.Y_test[batch_indices])






# Fashion-Mnist
# class FashionMnistData(DataModule):
#     def __init__(self, batch_size = 10, root='../dataset'):
#         super().__init__()
#         self.root = root
#         self.batch_size = batch_size
#         self.X_train = np.load(root+"/train_x.npy").transpose()
#         self.X_test = np.load(root+"/test_x.npy").transpose()

#         self.Y_train= np.load(root + "/train_y.npy").transpose()
#         self.Y_test = np.load(root+"/test_y.npy").transpose()
#         self.X_train = self.X_train.reshape(self.X_train.shape[0], 1, 28, 28)
#         self.X_test = self.X_test.reshape(self.X_test.shape[0], 1, 28, 28)

#         self.num_train = self.X_train.shape[0]
#         self.num_val = self.Y_train.shape[0]

#     def get_dataloader(self,train=True):
#         if train:
#             indices = list(range(self.num_train))
#             random.shuffle((indices))
#             for i in range(0, len(indices), self.batch_size):
#                 batch_indices = indices[i: min(i + self.batch_size,self.num_train)]
#                 yield self.X_train[batch_indices], self.Y_train[batch_indices]
#         else:
#             indices = list(range(self.num_val))
#             for i in range(0, len(indices), self.batch_size):
#                 batch_indices = indices[i: min(i + self.batch_size,self.num_val)]
#                 yield self.X_test[batch_indices], self.Y_test[batch_indices]

# if DBGFashionMnist:
#     fashion_mnist=FashionMnistData()
#     print(next(fashion_mnist.get_dataloader(train=True))[0].shape)
#     """ data shape
#     [batch_size, channels=1, height,width] -- [10, 1, 28, 28]
#     """


# class SVHNData(DataModule):
    # def __init__(self,batch_size =10,root="../dataset"):
    #     """
    #     Load the dataset
    #     :param batch_size:
    #     :param root:Where to locate the data
    #     """
    #     super().__init__()
    #     Mat_train =loadmat(root+"/train.mat")
    #     Mat_test = loadmat(root+"/test.mat")
    #     train_data, train_labels = Mat_train['X'], Mat_train['y']            # train_data :(32, 32, 3, 73257) train_label :(73257, 1)
    #     test_data, test_labels = Mat_test['X'], Mat_test['y']                # test_data  :(32, 32, 3, 26032) test_label  :(26032, 1)

    #     self.num_train = train_data.shape[3]
    #     self.num_val = test_data.shape[3]

    #     self.X_train =np.transpose(train_data,(3,2,0,1))
    #     self.X_test =np.transpose(test_data,(3,2,0,1))

    #     self.Y_train =self.encode_onehot(train_labels)
    #     self.Y_test = self.encode_onehot(test_labels)

    #     self.batch_size =batch_size



    # def encode_onehot(self,y):
    #     """
    #     :param y: shape:[num_samples,1] the original label of each image
    #     :return:
    #         labels: shape [num_samples,10]
    #     """
    #     num_samples = y.shape[0]
    #     labels = np.zeros(shape=(num_samples,10))
    #     for index,y_i in enumerate(y):
    #         label_value = y_i[0]
    #         if label_value==10:
    #             labels[index][0]=1
    #         else:
    #             labels[index][label_value] = 1

    #     return labels


    # def get_dataloader(self,train=True):
    #     if train:
    #         indices = list(range(self.num_train))
    #         random.shuffle((indices))
    #         for i in range(0, len(indices), self.batch_size):
    #             batch_indices = indices[i: min(i + self.batch_size,self.num_train)]
    #             yield self.X_train[batch_indices], self.Y_train[batch_indices]
    #     else:
    #         indices = list(range(self.num_val))
    #         for i in range(0, len(indices), self.batch_size):
    #             batch_indices = indices[i: min(i + self.batch_size,self.num_val)]
    #             yield self.X_test[batch_indices], self.Y_test[batch_indices]
