
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

import load_data


def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Convolution Layer
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling)
    conv3 = maxpool2d(conv3, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Path and Parameters
title = 'Sex'
N_TRAIN = 2000
N_TEST = 454
BATCH_SIZE = 50
DROPOUT = 0.5
TRAIN_EPOCH = 100
LEARNING_RATE = 0.001
TRAIN_BATCH = N_TRAIN/BATCH_SIZE
TEST_BATCH = N_TEST/BATCH_SIZE
N_CLASS = 2
IMAGE_SIZE = [256, 256]

proj_dir = '/Users/qihucn/Documents/EE577/Project/super_face'
train_image_dir = os.path.join(proj_dir, 'data', '2D_face_256', 'train')
test_image_dir = os.path.join(proj_dir, 'data', '2D_face_256', 'test')
label_path = os.path.join(proj_dir, 'data', 'TDFN-Export (2014-08-20).txt')

print 'Load data ...'
train_ids, train_images, train_labels = load_data.read_image_label(train_image_dir, label_path, title)
test_ids, test_images, test_labels = load_data.read_image_label(test_image_dir, label_path, title)
train_images = np.asarray(train_images)
test_images = np.asarray(test_images)

# One-hot coding
train_labels_onehot = np.zeros((N_TRAIN, N_CLASS), dtype=int)
test_labels_onehot = np.zeros((N_TEST, N_CLASS), dtype=int)
for i in range(N_TRAIN):
    train_labels_onehot[i, int(train_labels[i])-1] = 1
for i in range(N_TEST):
    test_labels_onehot[i, int(test_labels[i])-1] = 1


x = tf.placeholder(tf.float32, [None, IMAGE_SIZE[0], IMAGE_SIZE[1]])
y = tf.placeholder(tf.float32, [None, N_CLASS])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

# Store layers weight & bias
weights = {
    # 3x3 conv, 1 input, 128 outputs
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 128])),
    # 3x3 conv, 128 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([3, 3, 128, 64])),
    # 3x3 conv, 64 inputs, 32 outputs
    'wc3': tf.Variable(tf.random_normal([3, 3, 64, 32])),
    # fully connected,
    'wd1': tf.Variable(tf.random_normal([32*32*32, 200])),

    'out': tf.Variable(tf.random_normal([200, N_CLASS]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([128])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bc3': tf.Variable(tf.random_normal([32])),
    'bd1': tf.Variable(tf.random_normal([200])),
    'out': tf.Variable(tf.random_normal([N_CLASS]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

Train_ind = np.arange(N_TRAIN)
Test_ind = np.arange(N_TEST)

with tf.Session() as sess:
    sess.run(init)
    print 'Train ...'
    for epoch in range(0, TRAIN_EPOCH):

        Total_test_loss = 0
        Total_test_acc = 0

        for train_batch in range(0, TRAIN_BATCH):
            sample_ind = Train_ind[train_batch * BATCH_SIZE:(train_batch + 1) * BATCH_SIZE]
            batch_x = train_images[sample_ind, :, :]
            batch_y = train_labels_onehot[sample_ind, :]
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                           keep_prob: DROPOUT})

            if train_batch % BATCH_SIZE == 0:
                # Calculate loss and accuracy
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                                  y: batch_y,
                                                                  keep_prob: 1.})

                print("Epoch: " + str(epoch+1) + ", Batch: " + str(train_batch) + ", Loss= " + \
                      "{:.3f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(acc))

        # Calculate test loss and test accuracy
        print 'Test ...'
        for test_batch in range(0, TEST_BATCH):
            sample_ind = Test_ind[test_batch * BATCH_SIZE:(test_batch + 1) * BATCH_SIZE]
            batch_x = test_images[sample_ind, :]
            batch_y = test_labels_onehot[sample_ind, :]
            test_loss, test_acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                                        y: batch_y,
                                                                        keep_prob: 1.})
            Total_test_loss += test_loss
            Total_test_acc += test_acc

        Total_test_acc /= TEST_BATCH
        Total_test_loss /= TEST_BATCH

        print("Epoch: " + str(epoch + 1) + ", Test Loss= " + \
              "{:.3f}".format(Total_test_loss) + ", Test Accuracy= " + \
              "{:.3f}".format(Total_test_acc))

plt.subplot(2, 1, 1)
plt.ylabel('Test loss')
plt.plot(Total_test_loss, 'r')
plt.subplot(2, 1, 2)
plt.ylabel('Test Accuracy')
plt.plot(Total_test_acc, 'r')


print "All is well"
plt.show()
