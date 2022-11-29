#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：MyTorch
@Author  ：尤敬斌
@Date    ：2022/11/13 15:51
'''


import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
import itertools



class Performance:

    def __init__(self, target, labels):
        """
        构造函数，将传入列表要转换为numpy数组
        :param target: 预测值
        :param labels: 真实值
        """
        if not isinstance(target, np.ndarray):
            self.target = np.array(target)
        else:
            self.target = target
        if not isinstance(labels, np.ndarray):
            self.labels = np.array(labels)
        else:
            self.labels = labels

    def fit(self, target, labels):
        """
        重新保存数据
        :param target: 预测值
        :param labels: 真实值
        :return:
        """
        self.target = np.array(target)
        self.labels = np.array(labels)


class RegressionPerformance(Performance):   # 回归的性能指标类

    def getMSE(self):
        return np.mean(np.square(self.target - self.labels))

    def getMAE(self):
        return np.mean(np.abs(self.target - self.labels))

    def getR2(self):
        return 1 - self.getMSE() / (np.var(self.target) + 1e-30)


class ClassificationPerformance(Performance):   # 分类的性能指标类

    Near0 = 1e-30  # 防止分母为零

    def __init__(self, target, labels, classes):
        """
        重写构造方法，新增混淆矩阵属性
        :param target: 预测值
        :param labels: 真实值
        :param classes: 类别数
        """
        super().__init__(target, labels)
        self.confusionMatrix = None
        self.classes = classes

    def fit(self, target, labels, classes):
        """
        重写fit方法,重新初始化参数
        :param target: 预测值
        :param labels: 真实值
        :param classes: 类别数
        """
        self.__init__(target, labels, classes)

    def getConfusionMatrix(self):
        """
        获取混淆矩阵
        :return: 混淆矩阵，二维的numpy数组
        """
        if self.confusionMatrix is None:
            # 初始化混淆矩阵
            self.confusionMatrix = np.zeros((self.classes, self.classes))

            for i in range(len(self.target)):
                predict = self.target[i]
                real = self.labels[i]
                self.confusionMatrix[predict][real] += 1
        return self.confusionMatrix

    def getAccuracy(self):
        """
        获取准确率
        :return: 准确率
        """

        return np.mean(np.equal(self.target, self.labels))

    def getRecall(self):
        """
        获取召回率
        :return: 所有类别的召回率
        """
        cm = self.getConfusionMatrix()
        recall_list = []
        for i in range(cm.shape[0]):
            recall = cm[i, i] / (np.sum(cm[i, :]) + ClassificationPerformance.Near0)
            recall_list.append(recall)
        return np.array(recall_list)

    def getPrecision(self):
        """
        获取精确率
        :return: 所有类别的精确率
        """
        cm = self.getConfusionMatrix()
        precision_list = []
        for i in range(cm.shape[1]):
            recall = cm[i, i] / (np.sum(cm[:, i]) + ClassificationPerformance.Near0)
            precision_list.append(recall)
        return np.array(precision_list)

    def getF1Score(self):
        """
        获取F1分数
        :return: 所有类别的F1分数
        """
        recall = self.getRecall()
        precision = self.getPrecision()
        return 2 * precision * recall / (precision + recall + ClassificationPerformance.Near0)


    def showConfusionMatrix(self):
        """
        显示混淆矩阵
        """
        self.confusionMatrix = self.getConfusionMatrix()
        self.plot_confusion_matrix(self.confusionMatrix, range(self.classes), normalize=True)


    def getClassificationReport(self):
        """
        获取分类报告，这里直接调用sklearn的接口
        :return: 分类报告
        """
        return classification_report(self.target, self.labels)


    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        """
        - cm : 计算出的混淆矩阵的值
        - classes : 混淆矩阵中每一行每一列对应的列
        - normalize : True:显示百分比, False:显示个数
        """
        if normalize:
            cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-20)
            print("显示百分比：")
            np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
            print(cm)
        else:
            print('显示具体数字：')
            print(cm)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        # matplotlib版本问题，如果不加下面这行代码，则绘制的混淆矩阵上下只能显示一半，有的版本的matplotlib不需要下面的代码，分别试一下即可
        plt.ylim(len(classes) - 0.5, -0.5)
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()