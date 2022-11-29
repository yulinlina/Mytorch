import unittest

from ..torch.mlp import MLP

class MLpTest(unittest.TestCase):

    def test_MLP(self):
       model =MLP()
       model.show()



if __name__ == '__main__':
    unittest.main()  # 运行所有的测试用例