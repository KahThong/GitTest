import tensorflow as tf

n_inputs = 28*28
n_hidden1 = 400
n_hidden2 = 200
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None),name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X,n_hidden1, name="hidden1",activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1,n_hidden2,name="hidden2",activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2,n_outputs,name="outputs")

with tf.name_scope("loss"):
    Xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=y)
    loss = tf.reduce_mean(Xentropy,name="loss")

learning_rate = 0.0001

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits,y,1)
    accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")
X_test = mnist.validation.images
y_test = mnist.validation.labels

n_epochs = 40
batch_size = 100

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for interation in range(mnist.train.num_examples//batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op,feed_dict={X:X_batch, y:y_batch})
        acc_train = accuracy.eval(feed_dict={X:X_batch, y:y_batch})
        acc_val = accuracy.eval(feed_dict={X:X_test, y:y_test})
        print("{} Train accuracy: {}, Val accuracy: {}".format(epoch,acc_train,acc_val))
    save_path = saver.save(sess,"./MNIST_with_DNN.ckpt")
