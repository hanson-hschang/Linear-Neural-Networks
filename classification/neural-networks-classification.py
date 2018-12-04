import tensorflow as tf
import numpy as np
from scipy.linalg import expm
from read_data import read_data, train_test_split, batch_feeder

data_X0, data_Z = read_data("data_file_0.json")
data_n = data_Z.shape[1]
train_size = 0.002
validate_size = 0.99
# X0 is 2-by-number of samples
# Z is 1-by-number of samples  
X0_train, Z_train, _, _, X0_test, Z_test = train_test_split(data_X0, data_Z, train_size = train_size, validate_size=validate_size)
train_n = Z_train.shape[1]  # number of trainning examples
test_n = Z_test.shape[1]    # number of testing examples 
dim_n = X0_train.shape[0]

batch_size = 20 # number of trainning examples in each batch (epoch)
N_epoch = 100
N_batch = train_n // batch_size
N_iteration = N_batch * N_epoch
learning_rate = 1e-1



# Placeholders for data
X0 = tf.placeholder(tf.float64, [dim_n, None])
Z = tf.placeholder(tf.float64, [1, None])

# Model parameters
beta = (np.random.rand(dim_n)-0.5)*2
beta = beta/np.linalg.norm(beta)

# Change Model parameters to tensorflow constant
beta_ = tf.constant(beta)

# Initialize starting values, could be random as well
A = {'A0':tf.Variable(np.random.rand(dim_n,dim_n),dtype=tf.float64),
      'A1':tf.Variable(np.random.rand(dim_n,dim_n),dtype=tf.float64),
      'A2':tf.Variable(np.random.rand(dim_n,dim_n),dtype=tf.float64)
      }

def forward_with_sigmoid(A, X0, beta):
    """Forward pass for our fuction"""

    # Layer 1 Computation
    X1 = tf.sigmoid(tf.matmul(tf.linalg.expm(A['A0']),X0))

    # Layer 2 Computation
    X2 = tf.sigmoid(tf.matmul(tf.linalg.expm(A['A1']),X1))
    
    # Layer 3 Computation
    X3 = tf.sigmoid(tf.matmul(tf.linalg.expm(A['A2']),X2))
    
    # Output Layer Computation
    XF = tf.sigmoid(tf.tensordot(beta, X3, 1))

    return XF, (X1, X2, X3)

def forward(A, X0, beta):
    """Forward pass for our fuction"""

    # Layer 1 Computation
    X1 = tf.matmul(tf.linalg.expm(A['A0']),X0)

    # Layer 2 Computation
    X2 = tf.matmul(tf.linalg.expm(A['A1']),X1)
    
    # Layer 3 Computation
    X3 = tf.matmul(tf.linalg.expm(A['A2']),X2)
    
    # Output Layer Computation
    XF = tf.sigmoid(tf.tensordot(beta, X3, 1))

    return XF, (X1, X2, X3)

# There will be no error here, but it does not always converge to the relation "w" <--------------------------------------------------------
XF, (X1, X2, X3) = forward_with_sigmoid(A, X0, beta_) 

# This will cause Error <-------------------------------------------------------------------------------------------------------------------
# XF, (X1, X2, X3) = forward(A, X0, beta_)


# Caused by op 'matrix_exponential_2/MatrixSolve', defined at:
#   File "neural-networks-classification.py", line 75, in <module>
#     XF, (X1, X2, X3) = forward(A, X0, beta_)
#   File "neural-networks-classification.py", line 67, in forward
#     X3 = tf.matmul(tf.linalg.expm(A['A2']),X2)
# InvalidArgumentError (see above for traceback): Input matrix is not invertible.

# Cost Function
def cross_entropy(Z, XF):
    """Compute cost"""
    Z = tf.reshape(Z, [-1])
    return -tf.reduce_mean(
                tf.add(
                    tf.multiply(Z,tf.math.log(XF)),
                    tf.multiply(tf.subtract(tf.constant(1.0, tf.float64),Z),
                                tf.math.log(tf.subtract(tf.constant(1.0, tf.float64),XF)))))

cost = cross_entropy(Z, XF)

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) 
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost) 

# Accuracy
accuracy = tf.reduce_mean(tf.cast(tf.equal(Z, tf.round(XF)), tf.float64))


train_cost_history = np.zeros(N_iteration)
test_cost_history = np.zeros(N_epoch)
test_accuracy_history = np.zeros(N_epoch)
with tf.Session() as sess:
    
    # Initialize all defined variables
    tf.global_variables_initializer().run()

    for i in range(N_epoch):
        for k, (X0_train_feeder, Z_train_feeder) in enumerate(batch_feeder(X0_train, Z_train, batch_size=batch_size)):
            _, train_cost = sess.run(fetches=[optimizer, cost], feed_dict={X0: X0_train_feeder, Z: Z_train_feeder})
            # train_cost_ = sess.run(fetches=cost, feed_dict={X0: X0_train, Z: Z_train})
            train_cost_history[i*N_batch+k] = train_cost
        
        test_cost, test_accuracy = sess.run(fetches=[cost, accuracy], feed_dict={X0: X0_test, Z: Z_test})
        test_cost_history[i] = test_cost
        test_accuracy_history[i] = test_accuracy
        
        print("Epoch: {:2}\tLoss: {:.3f}\tAcc: {:.2%}".format(i, test_cost, test_accuracy))
    
    output = sess.run(fetches=XF, feed_dict={X0: X0_test, Z: Z_test})
    final_A = sess.run(A)
    I = np.identity(dim_n)
    phi = I
    for key, Ai in final_A.items():
        phi = np.dot(expm(Ai),phi)
    w_hat = np.matmul(phi,beta)
    w_hat = w_hat / np.linalg.norm(w_hat)
    print("phi = %s" % phi)
    print("w_hat = %s" % w_hat)