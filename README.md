# deep learning note

### [李宏毅教授深度学习](http://speech.ee.ntu.edu.tw/~tlkagk/courses_MLSD15_2.html '课程地址')

#### [Backpropagation](http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2015_2/Lecture/DNN%20backprop.ecm.mp4/index.html '反向传播')

#### [Tips for Training Deep Neural Network](http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2015_2/Lecture/Deep%20More%20(v2).ecm.mp4/index.html '训练神经网络的提示')

#### [Neural Network with Memory](http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2015_2/Lecture/RNN%20(v4).ecm.mp4/index.html '记忆神经网络')

#### [Training Recurrent Neural Network](http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2015_2/Lecture/RNN%20training%20(v6).ecm.mp4/index.html '训练循环神经网络')

#### [Introduction of Structured Learning](http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2015_2/Lecture/Structured%20Introduction%20(v2).ecm.mp4/index.html '结构化学习介绍')

#### Parameter Initialization:
结合了[《解析卷积神经网络—深度学习实践手册》](http://lamda.nju.edu.cn/weixs/book/CNN_book.html '解析卷积神经网络—深度学习实践手册')，感谢作者魏秀参（Xiu-Shen WEI）

#### word2vec: 
参考论文Xin Rong的word2vec Parameter Learning Explained，以及李沐老师的[深度学习课程](http://zh.gluon.ai/chapter_natural-language-processing/index.html '李沐深度学习课程')

#### sigmoid_network，使用转换函数为sigmoid，[参考代码地址](https://github.com/mnielsen/neural-networks-and-deep-learning)

#### CNN For NLP
[Understanding Convolutional Neural Networks for NLP（译文）](http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/ '原文地址')<br />
[Implementing a CNN for Text Classification in TensorFlow（译文）](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/ '原文地址')
[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882 )


模型构成：
1、输入层为预训练的词向量层，将1-of-V编码为低维向量空间（V为词汇表大小），本质是特征检测器，将维度单词的语义特征进行编码。通过这种紧凑表示，在低维空间，语义相似的词在欧几里得或余弦距离都会很相近。假设一个句子有10个词，shape=(10, 1)，使用词向量后转为低维矩阵为shape=(10, 300)，这里将词汇表V大小降维为300维；
2、下一层使用不同（窗口）形状大小的卷积核对词向量进行卷积操作。一般为(2, 3, 4, 5)，即卷积核高度为(2, 3, 4, 5)，分别扫过两个词、三个词、四个词和5个词，类似于n-gram。卷积核宽度则为300，即低维矩阵的宽度，只使用卷积核高度来控制扫过的词数目。注意，因为每个句子长度不一样，需要使用padding将其填充为相同长度，即最大句子的长度；
3、根据卷积核的高度，扫过句子矩阵，会产生特征图（feature map）。然后使用最大池化操作，其思想是抓住最重要特征，即每个特征图的最大值。如使用4个卷积核，则会产生4个最大值，一个卷积核提取一个重要特征。这些特征组成倒数第二层。池池化操作可以处理不同的句子长度；
4、最后一层为全连接层，使用softmax，输出概率。

#### Weight Initialization
译成中文，原代码地址：https://github.com/udacity/deep-learning/tree/master/weight-initialization 
