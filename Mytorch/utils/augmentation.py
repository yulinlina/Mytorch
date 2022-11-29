import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# shape: (sample_nums, 784) value : 0->1

"""传入训练集所有图像，work() 方法返回增广后的数据"""
class Augmentation:
    def __init__(self, images):
        self.nums = images.shape[0]
        self.img = images.copy()

    def filp_left_right(self, idx):
        h, w = 28, 28
        image = self.img[idx].copy()
        image = image.reshape(h, w)
        for j in range(0, h // 2):
            for i in range(0, h):
                image[i][j], image[i][w - 1 - j] = image[i][w - 1 - j], image[i][j]
        image = image.reshape(h * w)
        return image

    def file_up_down(self, idx):
        h, w = 28, 28
        image = self.img[idx].copy()
        image = image.reshape(h, w)
        for i in range(0, h // 2):
            for j in range(0, w):
                image[i][j], image[h - 1 - i][j] = image[h - 1 - i][j], image[i][j]
        image = image.reshape(h * w)
        return image

    def move(self, idx):
        """随机选择一个方向移动随机距离"""
        h, w = 28, 28
        image = self.img[idx].copy()
        image = image.reshape(h, w)
        dir = np.random.randint(4)  # 随机一个方向
        dist = np.random.randint(min(h // 19, w // 19)) + 1  # 随机 [1, min(w // 4, h // 4)] 距离
        if dir == 0:    # 向下
            for i in range(h - 1, dist - 1, -1):
                image[i] = image[i - dist]
            for i in range(dist):
                image[i] = np.zeros(w)
        elif dir == 1:  # 向上
            for i in range(0, h - dist):
                image[i] = image[i + dist]
            for i in range(h - dist, h):
                image[i] = np.zeros(w)
        elif dir == 2:  # 向右
            for j in range(w - 1, dist - 1, -1):
                for i in range(0, h):
                    image[i][j] = image[i][j - dist]
            for j in range(dist):
                for i in range(h):
                    image[i][j] = 0
        elif dir == 3:  # 向左
            for j in range(0, w - dist):
                for i in range(0, h):
                    image[i][j] = image[i][j + dist]
            for j in range(w - 1, w - 1 - dist, -1):
                for i in range(h):
                    image[i][j] = 0
        image = image.reshape(h * w)
        return image

    def make_noise(self, idx):
        h, w = 28, 28
        image = self.img[idx].copy()
        image = image.reshape(h, w)
        for i in range(h):
            for j in range(w):
                p = np.random.rand(1)
                if p < 0.05:    # 5 % 噪点
                    image[i][j] = np.float(np.random.rand(1))
        image = image.reshape(h * w)
        return image

    def show(self, image):
        h, w = 28, 28
        image = image.reshape(h, w)
        plt.imshow(image)
        plt.show()

    def work(self):
        for i in range(self.nums):
            image = self.img[i]
            dir = np.random.randint(4)
            if dir == 0:
                image = self.make_noise(i)
            elif dir == 1:
                image = self.file_up_down(i)
            elif dir == 2:
                image = self.filp_left_right(i)
            else:
                image = self.move(i)
            self.img[i] =image

    def diff(self):
        idx = np.random.randint(self.nums)
        pic = self.img[idx].copy()
        print("原图像：")
        self.show(pic)
        print("加入噪声：")
        self.show(self.make_noise(idx))
        print("平移：")
        self.show(self.move(idx))
        print("左右翻转：")
        self.show(self.filp_left_right(idx))
        print("上下翻转：")
        self.show(self.file_up_down(idx))


