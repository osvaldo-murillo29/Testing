import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import os

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)


def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    #print(tf.Variable)
    return tf.Variable(weights)

def forwardprop(X, w_1, w_2):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    """
    h    = tf.nn.sigmoid(tf.matmul(X, w_1))  # The \sigma function
    yhat = tf.matmul(h, w_2)  # The \varphi function
    return yhat

def get_data():
    """ Read qbo sender data and start training """

    qbo_topic_echo   = np.genfromtxt('training2.1.csv', delimiter=',')
    data   = qbo_topic_echo[:,[0,1,2]]
    target =  np.divide(qbo_topic_echo[:,[3]], qbo_topic_echo[:,[4]])

    print("3: ",qbo_topic_echo[:,[3]])
    print("4: ",qbo_topic_echo[:,[4]])

    print('\n',target)


    # Prepend the column of 1s for bias
    N, M  = data.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = data

    #print(data)

    all_Y = target
    return train_test_split(all_X, all_Y, test_size=0.13, random_state=RANDOM_SEED)

def main():
    train_X, test_X, train_y, test_y = get_data()



    p_keep_conv = tf.placeholder("float")
    p_keep_hidden = tf.placeholder("float")


    # Layer's sizes
    x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
    #print(x_size)
    h_size = 10
    #print(h_size)               # Number of hidden nodes
    y_size = train_y.shape[1]   # Number of outcomes (3 iris flowers)
    #print(y_size)
    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    # Forward propagation
    yhat    = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, axis=1)

    # Backward propagation
    cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    updates = tf.train.RMSPropOptimizer(0.000001, 0.9).minimize(cost)






    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    #print(init)



    for epoch in range(5):
        # Train with each example
        for i in range(len(train_X)):
            #print(train_X)
            sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1],p_keep_conv: 0.8, p_keep_hidden: 0.5})
            #print(updates)
            #print(train_X[i: i + 1])
            print(sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1],p_keep_conv: 0.8, p_keep_hidden: 0.5}))

            #np.mean calcula la media

        train_accuracy = np.mean(np.argmax(train_y, axis=1) == sess.run(predict, feed_dict={X: train_X, y: train_y, p_keep_conv: 1.0, p_keep_hidden: 1.0}))
        #print(sess.run(predict, feed_dict={X: train_X, y: train_y, p_keep_conv: 1.0, p_keep_hidden: 1.0}))
        test_accuracy  = np.mean(np.argmax(test_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: test_X, y: test_y,
                                          p_keep_conv: 1.0,
                                          p_keep_hidden: 1.0}))


        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
              % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

    # save the model

    saver= tf.train.Saver()
    saver.save(sess, "NeuralNetwork/sufre")



    sess.close()


if __name__ == '__main__':
    main()
