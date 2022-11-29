from collections import OrderedDict


class Parameter:
    def __init__(self):
        self.mode = True    # 模式, 训练时为True,测试时为False
        self.w_dict = OrderedDict()
        self.b_dict = OrderedDict()
        self.z_dict = OrderedDict()
        self.a_dict = OrderedDict()

        # 先不要删下面这个，以后可能用的到
        self.delta_dict = OrderedDict()
        self.w_grad_dict = OrderedDict()
        self.b_grad_dict = OrderedDict()
        self.activation_function_dict = OrderedDict()


    def train(self):
        """
        进入训练模式
        :return:
        """
        self.mode = True
        return self.mode

    def eval(self):
        """
        进入测试模式
        :return:
        """
        self.mode = False
        return self.mode

    def update_w(self, layer_name, learning_rate):
        """
        更新权值w
        :param layer_name: 要更新的那一层的名字
        :param learning_rate: 学习率
        :return: 
        """
        assert self.mode == True, "测试模式下不允许更新w"
        self.w_dict[layer_name] -= learning_rate*self.w_grad_dict[layer_name]

    def update_b(self, layer_name, learning_rate):
        """
        更新偏置b
        :param layer_name: 要更新的那一层的名字
        :param learning_rate: 学习率
        :return:
        """
        assert self.mode == True, "测试模式下不允许更新b"
        self.b_dict[layer_name] -= learning_rate*self.b_grad_dict[layer_name]

    def add(self, other):
        """
        更新参数信息
        :param other: Parameter类型的对象
        :return:
        """
        self.w_dict.update(other.w_dict)
        self.b_dict.update(other.b_dict)
        self.z_dict.update(other.z_dict)
        self.a_dict.update(other.a_dict)
        self.delta_dict.update(other.delta_dict)
        self.w_grad_dict.update(other.w_grad_dict)
        self.b_grad_dict.update(other.b_grad_dict)
        self.activation_function_dict.update(other.activation_function_dict)
