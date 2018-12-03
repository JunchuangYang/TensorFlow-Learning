# *-* coding=utf-8 *-*
#__author__ = 'lenovo'


from tensorflow.examples.tutorials.mnist import  input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
# x不是一个特定的值，而是一个占位符placeholder
#这个张量的形状是[None，784 ]。（这里的None表示此张量的第一个维度可以是任何长度的。）
x = tf.placeholder(tf.float32,[None,784])
# 在这里，我们都用全为零的张量来初始化W和b因为我们要学习W和b的值，它们的初值可以随意设置。
#注意，W的维度是[784，10]，因为我们想要用784维的图片向量乘以它以得到一个10维的证据值向量，
# 每一位对应不同数字类。b的形状是[10]，所以我们可以直接把它加到输出上面。
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
# y 是我们预测的概率分布, y' 是实际的分布
# 实现模型, y是预测分布
y = tf.nn.softmax(tf.matmul(x,w) + b)
# 训练模型，y_是实际分布
y_ = tf.placeholder("float",[None,10])
#为了计算交叉熵，我们首先需要添加一个新的占位符用于输入正确值：
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
# 我们要求TensorFlow用梯度下降算法（gradient descent algorithm）以0.01的学习速率最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# 在运行计算之前，我们需要添加一个操作来初始化我们创建的变量
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
# 让模型循环训练1000次！
for i in range(1,1000):
    batch_xs , batch_ys = mnist.train.next_batch(100)
    sess.run(train_step , feed_dict={x:batch_xs,y_:batch_ys})

# 验证正确率
correct_prediction = tf.equal(tf.argmax(y,1),tf.arg_max(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))
