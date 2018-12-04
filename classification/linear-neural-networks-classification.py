import tensorflow as tf
import numpy as np
from scipy.linalg import expm
from read_data import read_data, train_test_split, batch_feeder

data_X0, data_Z = read_data("data_file_1.json")
data_n = data_Z.shape[1]
train_size = 0.7
X0_train, Z_train, _, _, X0_test, Z_test = train_test_split(data_X0, data_Z, train_size = train_size, validate_size=0)
train_n = Z_train.shape[1]  # number of trainning examples
test_n = Z_test.shape[1]    # number of testing examples 
dim_n = X0_train.shape[0]

batch_size = 20 # number of trainning examples in each batch (epoch)
N_epoch = 5
N_batch = train_n // batch_size
N_iteration = N_batch * N_epoch
learning_rate = 5e-5

print('\n\n\n\nTrain with', train_n,'samples. Feed with', batch_size,'samples at a time.', N_epoch, 'epoch(s) in total.')

# Model parameters

# beta = np.zeros(dim_n)
# beta[0] = 0.5
beta = (np.random.rand(dim_n)-0.5)*2
beta = beta/np.linalg.norm(beta)
print("beta = ", beta)

def exp_lost(Z, XF):
    """Compute cost"""
    Z = tf.reshape(Z, [-1])
    return tf.reduce_mean(tf.exp(-tf.multiply(Z,XF)))

def cross_entropy(Z, XF):
    """Compute cost"""
    Z = tf.reshape(Z, [-1])
    return -tf.reduce_mean(
                tf.add(
                    tf.multiply(Z,tf.math.log(XF)),
                    tf.multiply(tf.subtract(tf.constant(1.0, tf.float64),Z),
                                tf.math.log(tf.subtract(tf.constant(1.0, tf.float64),XF)))))
    
def forward_approxi(A, X0, beta):
    """Forward pass for our fuction"""
    I = tf.constant(np.identity(dim_n))

    # Layer 1 Computation
    X1 = tf.matmul(tf.add(I,A['A0']),X0)

    # Layer 2 Computation
    X2 = tf.matmul(tf.add(I,A['A1']),X1)
    
    # Layer 3 Computation
    X3 = tf.matmul(tf.add(I,A['A2']),X2)
    
    # Output Layer Computation
    XF = tf.tensordot(beta, X3, 1)

    XF = tf.nn.sigmoid(XF)

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
    XF = tf.tensordot(beta, X3, 1)

    XF = tf.nn.sigmoid(XF)

    return XF, (X1, X2, X3)


# Placeholders for data
X0 = tf.placeholder(tf.float64, [dim_n, None])
Z = tf.placeholder(tf.float64, [1, None])

# Change Model parameters to tensorflow constant
beta_ = tf.constant(beta)

# just some starting value, could be random as well
A = {'A0':tf.Variable(np.random.rand(dim_n,dim_n),dtype=tf.float64),
      'A1':tf.Variable(np.random.rand(dim_n,dim_n),dtype=tf.float64),
      'A2':tf.Variable(np.random.rand(dim_n,dim_n),dtype=tf.float64)
      }

# Model definition
# XF, (X1, X2, X3) = forward(A, X0, beta_)
XF, (X1, X2, X3) = forward_approxi(A, X0, beta_)

# Optimizer
# cost = exp_lost(Z, XF)
cost = cross_entropy(Z, XF)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) 

train_cost_history = np.zeros(N_iteration)
test_cost_history = np.zeros(N_iteration)
XF_history = np.zeros(N_iteration)
# First, we need to create a Tensorflow session object
with tf.Session() as sess:
    
    # Initialize all defined variables
    tf.global_variables_initializer().run()

    for i in range(N_epoch):
        for k, (X0_train_feeder, Z_train_feeder) in enumerate(batch_feeder(X0_train, Z_train, batch_size=batch_size)):
            _, train_cost, current_XF = sess.run(fetches=[optimizer, cost, XF], feed_dict={X0: X0_train_feeder, Z: Z_train_feeder})
            # train_cost_ = sess.run(fetches=cost, feed_dict={X0: X0_train, Z: Z_train})
            test_cost = sess.run(fetches=cost, feed_dict={X0: X0_test, Z: Z_test})
            
            train_cost_history[i*N_batch+k] = train_cost
            test_cost_history[i*N_batch+k] = test_cost
            # XF_history[i*N_batch+k] = current_XF
    
    # output = sess.run(fetches=XF, feed_dict={X0: X0_test, Z: Z_test})
    # print(output.shape)
    # for i, _ in enumerate(output):
    #     print('predict =', output[i], ';\tlabel = ', Z_test[0,i])

    # final_A = sess.run(A)
    # I = np.identity(dim_n)
    # phi = I
    # for key, Ai in final_A.items():
    #     print("%s = %s" % (key, Ai))
    #     # phi = np.dot(np.add(I,Ai),phi)
    #     phi = np.dot(expm(Ai),phi)
    #     print("phi_t_0 = %s" % phi)
    # w_hat = np.matmul(phi,beta)/2
    # print("phi = %s" % phi)
    # print("w_hat = %s" % w_hat)

print(train_cost_history)
print(XF_history)

import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

fontsize = 20
fig, ax = plt.subplots(1,1, figsize=(9,7))
ax.semilogy(train_cost_history, label='train')
ax.semilogy(test_cost_history, label='test')
ax.legend(fontsize = fontsize-5)
ax.tick_params(labelsize=fontsize)
ax.set_xticks(np.arange(0,N_iteration+N_batch,N_batch))
# ax.set_xticklabels(np.arange(0,N_epoch+1, N_epoch))
ax.set_ylim([0.01,5.1])
ax.set_xlabel('Iteration',fontsize = fontsize)
ax.set_ylabel('Cost',fontsize = fontsize)
ax.set_title('Linear Cont.-Time Neural Network: Binary Classification',fontsize = fontsize+2)
# plt.text(N_iteration*0.3,1.8,r'$J=E\left[\ \frac{1}{2}|X_T-Z|^2\right]$',fontsize = fontsize)
# plt.text(N_iteration*0.5,0.6,r'$R=$',fontsize = fontsize)
# plt.text(N_iteration*0.6,0.5,' {}  {}\n {}  {}'.format(R[0,0],R[0,1],R[1,0],R[1,1]),fontsize = fontsize)
# plt.text(N_iteration*0.5,0.25,r'$\hat R=$',fontsize = fontsize)
# plt.text(N_iteration*0.6,0.2,' {:.2f}  {:.2f}\n {:.2f}  {:.2f}'.format(R_hat[0,0],R_hat[0,1],R_hat[1,0],R_hat[1,1]),fontsize = fontsize)

fig, ax = plt.subplots(1,1, figsize=(9,7))
ax.scatter(X0_train[0,:],X0_train[1,:],c=Z_train[0,:])
# ax.arrow(0, 0, w[0], w[1], head_width=0.05, head_length=0.1, fc='k', ec='k')
plt.show()