---
sort: 5
---

# Mytorch.data

## 简介
`DataModule`类是数据的基类。通常使用`__init__`方法来准备数据。如果需要，这包括下载和预处理。  
`train_dataloader` 返回训练数据集的数据加载器。数据加载器是一个生成器，它在每次使用时产生一个数据批处理。  
`val_dataloader` 用于返回验证数据集加载器。
## 原理
核心函数为`get_dataloader` ,它将数据打乱并每次返回一个批量大小的数据。
``` python
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
``` 
## 读取Small_Mnist数据集

``` python
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
            self.X_train = trainData.reshape(self.num_train, 1,28,28)
            self.X_test = testData.reshape(self.num_val, 1,28,28)

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
```