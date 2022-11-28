---
sort: 9 
---

# Trainer
在trainer 封装了所有训练过程  
``` python
from __future__ import print_function
import time
import pickle
import matplotlib.pyplot as plt
import numpy as np


from ..loss.lossFunction import CrossEntropy
from ..optim.sgd import SGD



class Trainer:
    def __init__(self, max_epochs=5,gradient_clip_val=0):
        self.max_epochs =max_epochs
        self.total_loss =[]
        self.train_acc =[]
        self.test_acc = []

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader
        self.val_dataloader = data.val_dataloader
        self.num_train = data.num_train
        self.num_val = data.num_val


    def prepare_model(self, model):
        self.model = model


    def draw(self):
        x = range(self.epoch)
        plt.clf()
        # plt.ion()
        plt.xlabel('Epoch')
        plt.xlim([0,self.max_epochs])
        plt.plot(x, self.total_loss, color='r',linestyle='-', label='train loss')
        plt.plot(x, self.train_acc, color='b', linestyle='--', label='train acc')
        plt.plot(x, self.test_acc, color='g',  linestyle='--', label='test acc')
        plt.legend()
        plt.show( )
        # plt.pause(1)
        # plt.close()




    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optimizer = SGD(model, lr=0.005)
        self.criterion = CrossEntropy()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0

        for self.epoch in range(self.max_epochs):
            self.epoch += 1                             # for drawing picture and print condition
            self.fit_epoch()
            if self.epoch%10 ==0:
                print(f"\rEpoch: {self.epoch}/{self.max_epochs}, loss: {self.total_loss[-1]:.3f}",
                      f" train_acc: {self.train_acc[-1]:.3f}",
                      f" test_acc: {self.test_acc[-1]:.3f}",
                      f" examples/sec:{(self.num_train) / (self.time_end - self.time_start):.1f} ",end="\n")
        self.draw()


    def fit_epoch(self):
        self.total_loss_all = 0
        self.train_total_acc = 0
        self.test_all_acc = 0
        self.time_start = time.time()

        for self.train_batch_idx ,(x,y) in enumerate(self.train_dataloader(),start=1):
            self.optimizer.zero_grad()
            output=self.model(x)
            loss =self.criterion(output,y)
            acc = self.accuracy(output, y)

            loss.backward()
            self.optimizer.step()
            self.total_loss_all+= loss.datas
            self.train_total_acc += acc

            print(f"\rEpoch: {self.epoch}, train_batch_id: {self.train_batch_idx}",
                  f"loss: {loss.datas/x.shape[0]:.3f},train_acc: {acc:.3f}",
                  flush=True,end="")


        self.time_end = time.time()
        self.total_loss.append(self.total_loss_all/self.num_train)
        self.train_acc.append(self.train_total_acc/self.train_batch_idx)
        self.eval()


    def eval(self):
        for self.val_batch_idx ,(x,y) in enumerate(self.val_dataloader(),start=1):
            output=self.model(x)
            acc = self.accuracy(output, y)
            self.test_all_acc += acc

            print(f"\rEpoch: {self.epoch},test_batch_id : {self.val_batch_idx},  test_acc: {acc:.3f}",
            end="",flush=True)



        self.test_acc.append(self.test_all_acc/self.num_val)

    def accuracy(self,y_hat, y):
        """

        :param y_hat:shape(N,C) for each sample ,it has to be a vector of size (C)
        :param y: shape (N,C)  target with one hot encoding
        :return:
        """

        self.each_batch_size = y.shape[0]
        idy_hat = np.argmax(y_hat, axis=1)
        idy = np.argmax(y, axis=1)
        acc = sum(idy == idy_hat) /self.each_batch_size
        return acc
    ``` 