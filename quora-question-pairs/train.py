import tensorflow as tf
import numpy as np
import pandas, tqdm, os
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

data = pandas.read_csv('../../../data/train.csv')
train = data[:int(len(data)*.9)]
test = data[len(train):]
train = train.fillna('empty')
test = test.fillna('empty')

try:
    trX = np.load('../../../save/trX.npy')
    trY = np.load('../../../save/trY.npy')
    teX = np.load('../../../save/teX.npy')
    teY = np.load('../../../save/teY.npy')
except:
    trX = np.array([np.concatenate((
        embeddings.encode(train['question1'][K]),
        embeddings.encode(train['question2'][K])
    )) for K in tqdm.tqdm(range(len(train['question1'])))])

    trY = np.array([one_hot(train['is_duplicate'][K]) for K in range(len(train['question1']))])

    teX = np.array([np.concatenate((
        embeddings.encode(test['question1'][K]),
        embeddings.encode(test['question2'][K])
    )) for K in tqdm.tqdm(range(len(train), len(data)))])

    teY = np.array([one_hot(test['is_duplicate'][K]) for K in range(len(train), len(data))])

    # Save the processed dataset
    np.save('../../../save/trX.npy', trX)
    np.save('../../../save/trY.npy', trY)
    np.save('../../../save/teX.npy', teX)
    np.save('../../../save/teY.npy', teY)

    print "Processed data is saved to disk."
    
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
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(150):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_input: 0.8, p_keep_hidden: 0.5})
        print(i, np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX,
                                                         p_keep_input: 1.0,
                                                         p_keep_hidden: 1.0})))

    saver.save(sess, '../../../save/model.ckpt')
