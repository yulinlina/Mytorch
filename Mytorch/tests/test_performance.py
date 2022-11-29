#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：MyTorch
@Author  ：尤敬斌
@Date    ：2022/11/13 15:51
'''

import unittest
from sklearn import metrics
import numpy as np

from ..evaluator.performance import RegressionPerformance, ClassificationPerformance


class PerformanceTest(unittest.TestCase):
    def test_regression(self):
        y = np.array([1, 1, 5])
        y_hat = np.array([2, 3, 5])
        rp = RegressionPerformance(y, y_hat)

        MSE = metrics.mean_squared_error(y, y_hat)
        MAE = metrics.mean_absolute_error(y, y_hat)
        R2 = metrics.r2_score(y, y_hat)

        mse = rp.getMSE()
        mae = rp.getMAE()
        r2 = rp.getR2()
        self.assertEqual(mse, MSE)  # add assertion here
        self.assertEqual(mae, MAE)  # add assertion here
        self.assertEqual(r2, R2)  # add assertion here

    def test_classification_confusionMatrix(self):
        y = np.array([0, 1, 2, 3, 0, 3, 2, 1, 0, 3, 2, 1])
        y_hat = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 3, 2, 1])
        cp = ClassificationPerformance(y, y_hat, 4)
        confusionMatrix = cp.getConfusionMatrix()
        # print(confusionMatrix)
        cp.showConfusionMatrix()
        self.assertEqual((confusionMatrix == metrics.confusion_matrix(y, y_hat)).all(), True)


    def test_classification_report(self):
        y = np.array([0, 1, 2, 3, 0, 3, 2, 1, 0, 3, 2, 1])
        y_hat = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 3, 2, 1])
        cp = ClassificationPerformance(y, y_hat, 4)

        self.assertEqual(cp.getAccuracy(), metrics.accuracy_score(y, y_hat))
        self.assertEqual((cp.getRecall() == metrics.recall_score(y, y_hat, average=None)).all(), True)
        self.assertEqual((cp.getPrecision() == metrics.precision_score(y, y_hat, average=None)).all(), True)
        self.assertEqual((cp.getF1Score() == metrics.f1_score(y, y_hat, average=None)).all(), True)

        print("\n分类报告: ")
        print(cp.getClassificationReport())


if __name__ == '__main__':
    unittest.main()
