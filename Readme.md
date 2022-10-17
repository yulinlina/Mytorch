# 编程范式
OOP
# 设计哪些模块
## 0. data
类： DataLoad 
功能： 随机把数据分割成训练集和测试集
把数据转为tensor
## 1. nn
抽象类：layer
实现类：
**Linear**
**conv2**
Batchnorm
Maxpool
dropout
**父类：module**
module.show 绘图
实现类： MLP
**自定义层数：Sequential**
激活函数

- **Relu**
- **Sigmoid**
- Softmax
## 2. loss 
MSE
Crossentropy
## 3. optim
### 抽象类 optimizer

- SGD
- Adam
## 4. tensor
实现自动梯度
计算tensor的梯度
保存tensor的梯度
## 5. evaluator
类： FP,TP,TN,PN四项 + 混淆矩阵
 

