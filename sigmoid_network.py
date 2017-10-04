#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
对于前向神经网络使用随机梯度下降算法，梯度用后向传播进行计算
'''
import random
import numpy as np

class Network(object):
    def __init__(self, sizes):
        '''
        size为列表，包含每层神经元的数目。举例，如果列表为[2, 3, 1]，为3层神经网络，
        其中第一层包含2个神经元，第二层包含3个神经元，最后一层1个神经元。偏差和权重是随机
        初始化，使用均值为0和方差为1的高斯分布。注意第一层假设为输入层，为了方便不对第一层
        的神经元设置偏差，因为偏差只用于计算后一层的输出
        '''
        self.num_layers = len(sizes)
        self.sizes = sizes
        # 不对第一层输入层设置偏差
        self.baises = [np.random.randn(y, 1) for y in sizes[1:]]
        # W(ji)*x+b，j为后一层的神经元数目，i为前一层的神经元数目，这样权重不用转置
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:1], sizes[1:])]

    def feedforward(self, a):
        # 计算W(ji)*x+b
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, train_data, epochs, mini_batch_size, eta, test_data=None):
        '''
        训练神经网络使用mini-batch的随机梯度下降。训练集为一系列元组(x, y)表示训练集的
        输入和期望谁出。如果有测试集，在每轮迭代后将对测试集进行验证，并打印出进度
        '''
        if test_data:
            n_test = len(test_data)
        n = len(train_data)
        for j in range(epochs):
            # 对数据进行随机，使得每轮的mini-batch尽量不同
            random.shuffle(train_data)
            # 将训练集样本按照mini_batch_size切分，mini_batch_size为10，则将训练集切成5000份，每份10个样本点
            mini_batches = [train_data[k: k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            # 每个mini_batch包含10个元组，元组形式为(x, y)，其中x为784(将28*28的像素铺开)，y为10，10个数字哑变量处理
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print('Epoch {0}: {1} / {2}'.format(j, self.evaulate(test_data), n_test))
            else:
                print('Epoch {0} complete'.format(j))

    def update_mini_batch(self, mini_batch, eta):
        '''
        使用反向传播对单一的mini batch运行梯度下降更新网络的权重和偏差
        '''
        # 初始化权重和偏差的大小
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # w(1) = w(0) - (eta/mini_batch) * (dC/dw) = w(0) - (eta/mini_batch)*delta_w
        # 梯度下降公式，对损失函数进行微分
        self.weights = [w - (eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        '''
        返回元组形式(nabla_b, nabla_w)，为损失函数的梯度，nabla_b和nabla_w是层层叠加列表，
        类似于self.biases和self.weights
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        # 列表形式，存储所有的激活点，层层叠加
        activations = [x]
        # 列表形式，存储所有的z向量，层层叠加
        # z向量的计算公式: 权重和样本点的内积加上偏差， z = wx + b
        zs = []
        # 正向传播，计算每层网络的激活函数的输入z和神经元的输出(通过转换函数后的输出)activation
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activation.append(activation)

        # 反向传播，Δz → Δa → Δy → ΔC，即∂C/∂w = ∂z/∂w * ∂C/∂z
        # ∂C/∂z = ∂C/∂y * ∂y/∂z，∂y/∂z是对激活函数进行微分，因为改变激活函数的输入z，会改变神经元的输出
        # 进而印象到网络的输出y值，sigmoid_prime(zs[-1]对最后一层的激活函数微分，这里激活函数为sigmoid函
        # 数，即1.0 / (1.0 + np.exp(-z))，微分为sigmoid(z) * (1-sigmoid(z))
        # ∂C/∂y对损失函数进行微分，这里损失函数为1/2*(pred_y - true_y)**2，所以微分为(pred_y - true_y)
        delta = self.cost_derivative(activation[-1], y) * sigmoid_prime(zs[-1])
        # 最后一层z=wa+b，∂z/∂b = 1，即∂C/∂b = ∂z/∂b * ∂C/∂z = ∂z/∂b * ∂C/∂y * ∂y/∂z = 1 * ∂C/∂y * ∂y/∂z
        # 其中∂C/∂y * ∂y/∂z为delta，即对损失函数进行微分*对激活函数进行微分
        nabla_b[-1] = 1 * delta
        # 最后一层z=wa+b，∂z/∂b = 1，即∂C/∂w = ∂z/∂w * ∂C/∂z = ∂z/∂w * ∂C/∂y * ∂y/∂z = a * ∂C/∂y * ∂y/∂z
        # 其中∂C/∂y * ∂y/∂z为delta，即对损失函数进行微分*对激活函数进行微分，a为经过激活函数转换后的值，activations[-2].T
        # activations[-2]为激活函数转换后的值，activations[-1]最后一层为y值，不需要通过激活函数转换
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # 变量l=1，表示神经元的最后一层，l=2则是倒数第二层，以此类推
        for l in range(2, self.num_layers):
            z = zs[-l]
            # 对激活函数进行微分
            sp = sigmoid_prime(z)
            # 假设反向传播求的梯度不是最后一层，计算第l层和第l+1层的关系，Δz → Δa → Δy → ΔC
            # ∂C / ∂z(l) = ∂a(l)/∂z(l) * SUM(∂z(l+1)/∂a(l) * ∂C/∂z(l+1))
            # 表示第l层的激活函数的输入会影响到神经元的输出a，一个神经元的输出a会改变第l+1层的所有
            # 神经元的输入∂z(l+1)，而所有神经元的输入会影响到损失函数的结果
            # sp即∂a(l)/∂z(l)即对激活函数进行微分，这里使用np.dot为矩阵形式，因为改变的是整层网络
            # SUM(∂z(l+1)/∂a(l) * ∂C/∂z(l+1))，其中z(l+1) = w(l+1)*a(l)+b，对其微分
            # 即∂z(l+1)/∂a(l)=w(l+1)，也就是self.weights[-l+1].transpose()
            # ∂C/∂z(l+1)为delta，即第l+1层的梯度，计算方法如之前计算最后一层梯度
            # ∂C/∂z(l+1) = ∂y/∂z * ∂C/∂y，求激活函数的微分和损失函数的微分，然后相乘
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            # ∂C / ∂b(l) = ∂z(l)/∂b(l) * ∂a(l)/∂z(l) * SUM(∂z(l+1)/∂a(l) * ∂C/∂z(l+1))
            # 其中∂z(l)= w(l)*a(l-1)+b(l)，所以∂z(l)/∂b(l)=1
            nabla_b[-l] = 1 * delta
            # ∂C / ∂w(l) = ∂z(l)/∂w(l) * ∂a(l)/∂z(l) * SUM(∂z(l+1)/∂a(l) * ∂C/∂z(l+1))
            # 其中∂z(l)= w(l)*a(l-1)+b(l)，所以∂z(l)/∂w(l)=a(l-1)，即activations[-l-1].transpose()
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)


    def evaluate(self, test_data):
        '''
        返回神经网络输出正确结果的测试输入的数量。 注意，神经网络的输出被假定为最终层中具有最高激活的神经元的指数
        pred_y = wx+b，使用softmax选择值最大的转为整数，判断是否与true_y一致
        '''
        test_results = [(np.argmax(self.feedforward(x)), y) for x, y in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        '''对损失函数1/2*(pred_y - true_y)**2进行微分'''
        return (output_activations - y)

def sigmoid(z):
    '''激活函数sigmoid函数'''
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    '''对激活函数sigmoid函数进行微分'''
    return sigmoid(z) * (1-sigmoid(z))
