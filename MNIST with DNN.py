import tensorflow as tf
import time

# code snippet to disable GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

n_inputs = 28*28
n_hidden1 = 400
n_hidden2 = 200
n_outputs = 10

# X =  input (image), Y = input (Solution), remember to specify shape for efficiency
# N_inputs = Size of image in terms of pixel
# Convert 28*28 into 1*784 array

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

# tf.layers.dense is a fully connected hidden layer and previous layer
# activation: relu eliminates useless data by removing x<0, only taking x>0, other activation includes relu, tanh
# one hot encoder (search this)
# dnn defines hidden layers

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

# Xentropy compares between Y and logits
with tf.name_scope("loss"):
    Xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=y)
    loss = tf.reduce_mean(Xentropy, name="loss")

learning_rate = 0.0001

# gradient descent (learning rate determine convergence rate/steps)
with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

# accuracy nn.in_top_k compares logits and y for accuracy
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()
# save the variables (all dnn elements)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")
# take in additional test data as unique sets
# test sets shldnt be seen in optimization
X_test = mnist.validation.images
y_test = mnist.validation.labels

# n_epochs is number of loops
n_epochs = 40
# RAM limitations/ prevent over fitting
batch_size = 100

start = time.time()

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for interation in range(mnist.train.num_examples//batch_size):
            # train set pull in
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            # new variables will always go back to the basic element, which in this case is X(placeholder)
            sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
        # do the same as above line, but compare with Y train instead
        acc_train = accuracy.eval(feed_dict={X:X_batch, y:y_batch})
        # do the same as above, but with X and Y test set
        acc_val = accuracy.eval(feed_dict={X:X_test, y:y_test})
        print("{} Train accuracy: {}, Val accuracy: {}".format(epoch,acc_train,acc_val))
    save_path = saver.save(sess,"./MNIST_with_DNN.ckpt")

end = time.time()
print(end - start)
