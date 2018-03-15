import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

def add_layer(inputs, input_size, output_size,n_layer, activation_function = None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([input_size,output_size]),name='W')
            tf.summary.histogram(layer_name+'/weights',Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,output_size])+0.1)
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs,Weights)+biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name + '/outputs', outputs)
        return outputs

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result


with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 784], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 10], name='y_input')


l1 = add_layer(xs, 784, 100, n_layer=1,activation_function=tf.nn.tanh)
prediction = add_layer(l1, 100, 10, n_layer=2,activation_function=tf.nn.softmax)
# prediction = add_layer(xs, 784, 10, n_layer=1,activation_function=tf.nn.softmax)
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))
    tf.summary.scalar('cross_entropy',cross_entropy)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    writer = tf.summary.FileWriter("E://", sess.graph) #保存文件到指定目录，用tenserboard打开
    sess.run(init)
    merge = tf.summary.merge_all()#merge所有的summary


    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
        if i % 50 == 0:
            result2 = sess.run(merge,feed_dict={xs:batch_xs,ys:batch_ys})
            writer.add_summary(result2,i)
            print(compute_accuracy(mnist.test.images,mnist.test.labels))



