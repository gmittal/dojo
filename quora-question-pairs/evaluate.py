import tensorflow as tf
import numpy as np
import pandas, tqdm, os, sys
import embeddings

os.chdir('/root/good-question/deduplication/util/InferSent/encoder')

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def one_hot(class_idx):
    return np.eye(2)[class_idx]

def model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden): # this network is the same as the previous one except with an extra hidden layer + dropout
    X = tf.nn.dropout(X, p_keep_input)
    h = tf.nn.relu(tf.matmul(X, w_h))

    h = tf.nn.dropout(h, p_keep_hidden)
    h2 = tf.nn.relu(tf.matmul(h, w_h2))

    h2 = tf.nn.dropout(h2, p_keep_hidden)

    return tf.matmul(h2, w_o)

embeddings.init(infersent=False, glove='glove.840B.300d.txt')

X = tf.placeholder("float", [None, 600])
Y = tf.placeholder("float", [None, 2])

w_h = init_weights([600, 450])
w_h2 = init_weights([450, 150])
w_o = init_weights([150, 2])

p_keep_input = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w_h, w_h2, w_o, p_keep_input, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

saver = tf.train.Saver()

# Launch the graph in a session
sess = tf.Session()
saver.restore(sess, '../../../save/model.ckpt')

def is_duplicate(question1, question2):
    return sess.run(predict_op, feed_dict={X: np.concatenate((embeddings.encode(question1), embeddings.encode(question2))).reshape(1, 600),
                                        p_keep_input: 1.0,
                                        p_keep_hidden: 1.0})[0] == 1

print is_duplicate(sys.argv[1], sys.argv[2])
