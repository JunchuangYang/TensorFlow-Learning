# *-* coding=utf-8 *-*
#__author__ = 'lenovo'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#定义一个函数，用于初始化所有的权值 W
def weight_variable(shape):
    # 生成一个截断的正态分布，标准差为0.1
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

#定义一个函数，用于初始化所有的偏置项 b
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#定义一个函数，用于构建卷积层
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

#定义一个函数，用于构建池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


def main():
    mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

    x = tf.placeholder(tf.float32,shape=[None,784]) #输入的数据占位符 28*28的图片转换为一维的向量
    y_ = tf.placeholder(tf.float32,shape=[None,10]) #输入的标签占位符
   
    # 改变x的格式转为4D的向量[batch,in_height,in_width,in_channels]
    x_image = tf.reshape(x,[-1,28,28,1])   
    # 初始化第一层卷积的权值和偏置值
    w_conv1 = weight_variable([5,5,1,32])#5*5的采样窗口，32个卷积核从1个平面抽取特征
    b_conv1 = bias_variable([32]) # 每一个卷积核一个偏置值

    # 把x_image和权值向量进行卷积，加上偏置值，然后用用户relu激活函数
    h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1) + b_conv1 )
    h_pool1 = max_pool_2x2(h_conv1) # 进行max—pooling

    # 第二层卷积
    # 第一层卷积中用了32个卷积核,所以向第二个卷积层输入时输入32个平面
    # 64个卷积核从32个平面中抽取特征
    w_conv2 = weight_variable([5,5,32,64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)


    # 28*28的图片第一次卷积后还是28*28（padding使用的是SAME，大小不会改变），
    # 第一次池化后变为14*14（2*2的窗口，步长为2）
    # 第二次卷积后为14*14，第二次池化后变为7*7
    # 通过上面的操作后得到了64张7*7的平面


    # 密集连接层
    # 初始化第一个全连接层的权值
    # 上一层有7*7*64个神经元，全连接层有1024个神经元
    w_fc1 = weight_variable([7*7*64,1024])
    b_fc1 = bias_variable([1024]) # 1024个偏置值

    # 把池化层2的输出扁平化为一维，-1 表示任意
    h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
    # 求第一个全连接层的输出
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1) + b_fc1)  #第一个全连接层

    # Dropout：部分神经元工作，部分神经元不工作，防止过拟合
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

    # 输出层
    # 初始化第二个全连接层
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    # 计算输出，转化为概率得到输出
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # 训练和评估模型
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv)) #交叉熵代价函数
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) #使用AdamOptimizer进行优化
    # 结果存放在一个bool列表中，argmax返回一维张量中最大值的所在位置
    correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float')) #精确度计算
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(20000):
            batch = mnist.train.next_batch(50)
            if i%100 ==0:  #训练100次，验证一次
                train_accuracy = accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
                print("step %d,training accuracy %g"%(i,train_accuracy))

            train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})

        print("test accuracy %g"%accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))

if __name__ == '__main__':
    main()

