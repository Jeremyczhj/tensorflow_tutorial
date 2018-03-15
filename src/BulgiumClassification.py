import tensorflow as tf
import os
from skimage import transform
from skimage import data
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
import random

def load_data(data_directory):
    directiories = [d for d in os.listdir(data_directory)  #Return a list containing the names of the files in the directory 找data_directory下的子目录名称，并放到list中
                    if os.path.isdir(os.path.join(data_directory,d))]#join之后相当于得到了Training目录下左右子目录的路径
    #print(directiories)
    labels = []
    images = []#初始化两个列表：labels 和 imanges
    for d in directiories:

        label_directory = os.path.join(data_directory,d)#data_directory是Training目录路径
        #print(label_directory)
        file_names = [os.path.join(label_directory,f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(data.imread(f))
            labels.append(int(d))
            #print(images)
    return images,labels


ROOT_PATH = "E:\EisleyChan\下载\学习资料\BelgiumTSC_TT" #设置你的 ROOT_PATH。这个路径是带有你的训练数据和测试数据的目录。
train_data_dir = os.path.join(ROOT_PATH,'Training') #借助 join() 函数为 ROOT_PATH 增加特定的路径。'Training'用""会报错
print(train_data_dir)
test_data_dir = os.path.join(ROOT_PATH,"Testing")
images,labels = load_data(train_data_dir)

images_array = np.array(images)
labels_array = np.array(labels)

print(images_array.ndim) # Print the `images` dimensions    1
print(images_array.size) # Print the number of `images`'s elements 4575
images_array[0]        # Print the first instance of `images`
print(labels_array.ndim) # Print the `labels` dimensions   1
print(labels_array.size)# Print the number of `labels`'s elements 4575
print(len(set(labels_array))) # Count the number of labels  62
#plt.hist(labels,62)
#plt.show()


# Determine the (random) indexes of the images
traffic_signs = [300, 2250, 3650, 4000]

# Fill out the subplots with the random images and add shape, min and max values
# for i in range(len(traffic_signs)):
#     plt.subplot(1, 4, i+1)
#     plt.axis('off')
#     plt.imshow(images[traffic_signs[i]])
#     plt.subplots_adjust(wspace=0.5)
#     plt.show()
#     print("shape: {0}, min: {1}, max: {2}".format(images[traffic_signs[i]].shape,
#                                                   images[traffic_signs[i]].min(),
#                                                   images[traffic_signs[i]].max()))

# unique_labels = set(labels)
# plt.figure(figsize=(15,15))
# i = 1
# for label in unique_labels:
#     image = images[labels.index(label)]
#     plt.subplot(8,8,i)#设置子图有几行几列
#     plt.axis('off')#关闭子图的坐标
#     # Add a title to each subplot
#     plt.title("Label {0} ({1})".format(label, labels.count(label)))
#     # Add 1 to the counter
#     i += 1
#     # And you plot this first image
#     plt.imshow(image)
# plt.show()

#resize images
images32 = [transform.resize(image,(28,28))for image in images]
images32 = np.array(images32)

for i in range(len(traffic_signs)):
    plt.subplot(1,4,i+1)
    plt.axis("off")
    plt.imshow(images32[traffic_signs[i]])
    plt.subplots_adjust(wspace=0.5)
    plt.show()
    print("shape:{0},min:{1},max:{2}".format(images32[traffic_signs[i]].shape,
                                             images32[traffic_signs[i]].min(),
                                             images32[traffic_signs[i]].max()))

#转换到灰度图
images32= rgb2gray(np.array(images32))
for i in range(len(traffic_signs)):
    plt.subplot(1,4,i+1)
    plt.axis("off")
    plt.imshow(images32[traffic_signs[i]],cmap="gray")
    plt.subplots_adjust(wspace=0.5)
#plt.show()
print(images32.shape)

x = tf.placeholder(dtype= tf.float32,shape = [None,28,28])
y = tf.placeholder(dtype=tf.int32,shape=[None])
images_flat = tf.contrib.layers.flatten(x)
layer1= tf.layers.dense(images_flat,512,tf.nn.relu)
layer2 = tf.layers.dense(layer1,256,tf.nn.relu)
logits = tf.layers.dense(layer2,62,tf.nn.relu)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits))
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_pred = tf.argmax(logits, 1)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print("images_flat: ", images_flat)
print("logits: ", logits)
print("loss: ", loss)
print("predicted_labels: ", correct_pred)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(200):
    print('epoch',i)
    _,accuracy_val = sess.run([train_op,accuracy],feed_dict={x:images32,y:labels})
    print("loss = ",loss)
    print('Done')

# Pick 10 random images
sample_indexes = random.sample(range(len(images32)), 10)
sample_images = [images32[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]

# Run the "predicted_labels" op.

predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]

# Print the real and predicted labels
print(sample_labels)
print(predicted)

# Display the predictions and the ground truth visually.
fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2,1+i)
    plt.axis('off')
    color='green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction),
             fontsize=12, color=color)
    plt.imshow(sample_images[i])
plt.show()


# Load the test data
test_images, test_labels = load_data(test_data_dir)

# Transform the images to 28 by 28 pixels
test_images28 = [transform.resize(image, (28, 28)) for image in test_images]

# Convert to grayscale
test_images28 = rgb2gray(np.array(test_images28))

# Run predictions against the full test set.
predicted = sess.run([correct_pred], feed_dict={x: test_images28})[0]

# Calculate correct matches
match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])

# Calculate the accuracy
accuracy = match_count / len(test_labels)

# Print the accuracy
print("Accuracy: {:.3f}".format(accuracy))