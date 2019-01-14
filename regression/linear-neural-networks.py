import tensorflow as tf
import numpy as np
from scipy.linalg import expm
import time
from data import *

class DataSet:
    """Data Generation

    read data from data file
    """

    def __init__(self, file_name="data_file.json", train_size=0.7, validation_size=0):
        self.file_name = file_name
        self.data_X0, self.data_Z = read_data(self.file_name)
        self.X0_train, self.Z_train, self.X_validation, self.Z_validation, self.X0_test, self.Z_test = \
            train_test_split(self.data_X0,self.data_Z, train_size = train_size, validation_size=0)
        self.train_m = self.Z_train.shape[1]                # number of trainning examples
        self.validation_m = self.Z_validation.shape[1]      # number of valadating examples
        self.test_m = self.Z_test.shape[1]                  # number of testing examples 
        self.dim_n = self.X0_train.shape[0]                 # input dimension

        print('\nTrained with', self.train_m,'samples, and tested with', self.test_m,'samples.\n')


class Model:
    """ Build Learning Model"""

    batch_size=70           # number of trainning examples in each batch (epoch)
    N_epoch=5               # number of epoch
    learning_rate=5e-2
    T=3
    l=0
    result=dict()

    def __init__(self, dataset):
        self.dataset = dataset
    
    def update_setting(self):                             
        self.N_batch = dataset.train_m // self.batch_size       # number of batch in the trainning set
        self.N_iteration = self.N_batch * self.N_epoch          # number of iterations

    def train(self, model_type='At'):
        self.update_setting()
        print('\nFeeded with', self.batch_size,'samples in a batch.', self.N_epoch, 'epoch(s) in total.')
        print('Learning Rate =', self.learning_rate,'. Number of inner layers =', self.T-1,'. Regularization parameter =', self.l,'.\n')

        # Placeholders for data
        X0 = tf.placeholder(tf.float64, [self.dataset.dim_n, None])
        Z = tf.placeholder(tf.float64, [self.dataset.dim_n, None])

        # Initializing Weights of Inner Layers
        def init_A(model_type):
            if model_type=='At':
                A = dict()
                for i in range(self.T):
                    A['A'+str(i)] = tf.Variable(np.random.rand(self.dataset.dim_n,self.dataset.dim_n))
            else:
                A = tf.Variable(np.random.rand(self.dataset.dim_n,self.dataset.dim_n))
            return A
        A = init_A(model_type)
        
        # Model Definition
        def forward(A, X0, model_type):
            Xi = [None]*self.T
            Xi[0] = X0
            for i in range(self.T-1):
                if model_type=='At':
                    Xi[i+1] = tf.matmul(tf.linalg.expm(A['A'+str(i)]),Xi[i])
                else:
                    A_transpose = tf.transpose(A)
                    Xi[i+1] = tf.matmul(tf.matmul(tf.linalg.expm(tf.cast(i+1,tf.float64)*(A-A_transpose)),
                                                    tf.linalg.expm(tf.cast(i+1,tf.float64)*A_transpose)),X0)
            XT = Xi[self.T-1]
            return XT, Xi
        XT, _ = forward(A, X0, model_type)

        # Cost Function
        cost = self.rmse(Z, XT) + self.regularization(self.l, A, model_type)

        # Optimizer
        # optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost) 
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost) 

        with tf.Session() as sess:
            
            print('Start Building Model ...')
            # Initialize all defined variables
            tf.global_variables_initializer().run()
            print('Model Initialized Completely ...')

            train_cost_history = np.zeros(self.N_iteration)
            test_cost_history = np.zeros(self.N_iteration)
            print('Start Trainning ...')
            for i in range(self.N_epoch):
                for k, (X0_train_feeder, Z_train_feeder) in enumerate(batch_feeder(X=self.dataset.X0_train, Z=self.dataset.Z_train, batch_size=self.batch_size)):
                    _, train_cost = sess.run(fetches=[optimizer, cost], feed_dict={X0: X0_train_feeder, Z: Z_train_feeder})
                    test_cost = sess.run(fetches=cost, feed_dict={X0: self.dataset.X0_test, Z: self.dataset.Z_test})
                    train_cost_history[i*self.N_batch+k] = train_cost
                    test_cost_history[i*self.N_batch+k] = test_cost
            print('Trainning Complete ...\n')
            
            self.result['train_cost'] = train_cost_history
            self.result['test_cost'] = test_cost_history
            self.result['A'] = sess.run(A)
            if model_type=='At':
                self.result['R_hat'] = np.identity(self.dataset.dim_n)
                for i in range(self.T):
                    self.result['R_hat'] = np.dot(expm(self.result['A']['A'+str(i)]),self.result['R_hat'])
            else:
                self.result['R_hat'] = np.matmul(expm(self.T*(self.result['A']-np.transpose(self.result['A']))),
                                                    expm(self.T*np.transpose(self.result['A'])))

        return self.result

    # Cost Function Definition
    def rmse(self, Z, Xn):
        """Compute root mean squared error"""
        return tf.sqrt(tf.reduce_mean(tf.square((Z - Xn))))
    
    def regularization(self, l, A, model_type):
        """Compute regularization cost"""
        if model_type=='At':
            trace_integral = 0
            for i in range(self.T):
                trace_integral += tf.reduce_sum(tf.square(A['A'+str(i)]))
        else:
            trace_integral = tf.reduce_sum(tf.square(A))
        return tf.cast(l, tf.float64)*trace_integral/tf.cast(2.,tf.float64)

if __name__ == '__main__':
    dataset = DataSet(file_name="exp_data_file.json")
    model = Model(dataset)
    model.l = 0
    model.N_epoch = 500
    result = model.train(model_type='A0')
    print(result['R_hat'])

    print('\nPlot trainning and testing costs ...\n')
    import matplotlib.pyplot as plt
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    fontsize = 20
    fig, ax = plt.subplots(1,1, figsize=(9,7))
    ax.semilogy(result['train_cost'], label=r'train')
    ax.semilogy(result['test_cost'], label=r'test')
    ax.legend(fontsize=fontsize-5)
    ax.tick_params(labelsize=fontsize)
    # ax.set_xticks(np.arange(0,model.N_iteration+model.N_batch,model.N_batch))
    # ax.set_ylim([0.001,5.1])
    ax.set_xlabel('Iteration', fontsize=fontsize)
    ax.set_ylabel('Cost', fontsize=fontsize)
    ax.set_title('Linear Cont.-Time Neural Network', fontsize=fontsize+2)
    # plt.text(N_iteration*0.3,1.8,r'$J=E\left[\ \frac{1}{2}|X_T-Z|^2\right]$', fontsize=fontsize)
    # plt.text(N_iteration*0.2,0.6,r'$R=$',fontsize=fontsize)
    # plt.text(N_iteration*0.3,0.5,' {}  {}\n {}  {}'.format(R[0,0],R[0,1],R[1,0],R[1,1]), fontsize=fontsize)
    # plt.text(N_iteration*0.5,0.6,r'$\hat R_{A_0}=$', fontsize=fontsize)
    # plt.text(N_iteration*0.65,0.5,' {:.2f}  {:.2f}\n {:.2f}  {:.2f}'.format(R_hat_A0[0,0],R_hat_A0[0,1],R_hat_A0[1,0],R_hat_A0[1,1]), fontsize=fontsize)
    # plt.text(N_iteration*0.5,0.25,r'$\hat R_{A_t}=$', fontsize=fontsize)
    # plt.text(N_iteration*0.65,0.2,' {:.2f}  {:.2f}\n {:.2f}  {:.2f}'.format(R_hat_At[0,0],R_hat_At[0,1],R_hat_At[1,0],R_hat_At[1,1]), fontsize=fontsize)
    # plt.text(N_iteration*0.3,0.08,'exec. time for $A_0$: {:.2f}+{:.2f} sec.,\n exec. time for $A_t$: {:.2f}+{:.2f} sec.'.format(init_A0, duation_A0, init_At, duation_At), fontsize=fontsize)
    plt.show()


# train_n = 10000   # number of trainning examples
# batch_size = 100 # number of trainning examples in each batch (epoch)
# N_batch = train_n // batch_size
# N_epoch = 5
# N_iteration = N_batch * N_epoch
# test_n = 100    # number of testing examples
# learning_rate = 5e-2

# # Model parameters
# dim_n = 2
# R = np.array([[0., -1.],
#               [1.,  0.]])
# l = 0.2
# noise_amp = 0.01

# def rmse(Z, Xn):
#     """Compute root mean squared error"""
#     return tf.sqrt(tf.reduce_mean(tf.square((Z - Xn))))

# def regularization(l, A):
#     """Compute regularization cost"""
#     trace_int = tf.reduce_sum(tf.square(A['A0'])) + tf.reduce_sum(tf.square(A['A1'])) + tf.reduce_sum(tf.square(A['A2']))
#     return l*trace_int/tf.cast(2.,tf.float64)


# def forward_approxi(A, X0):
#     """Forward pass for our fuction"""
#     I = tf.constant(np.array([[1.,0.],[0.,1.]]))

#     # Layer 1 Computation
#     X1 = tf.matmul(tf.add(I,A['A0']),X0)

#     # Layer 2 Computation
#     X2 = tf.matmul(tf.add(I,A['A1']),X1)
    
#     # Layer 3 Computation
#     X3 = tf.matmul(tf.add(I,A['A2']),X2)
    
#     return X3, (X1, X2)

# def forward(A, X0):
#     """Forward pass for our fuction"""

#     # Layer 1 Computation
#     X1 = tf.matmul(tf.linalg.expm(A['A0']),X0)

#     # Layer 2 Computation
#     X2 = tf.matmul(tf.linalg.expm(A['A1']),X1)
    
#     # Layer 3 Computation
#     X3 = tf.matmul(tf.linalg.expm(A['A2']),X2)
    
#     return X3, (X1, X2)


# def A0_forward(A0, X0, T):
#     A0_transpose = tf.transpose(A0)
#     XT = tf.matmul(tf.matmul(tf.linalg.expm(T*(A0-A0_transpose)),tf.linalg.expm(T*A0_transpose)),X0)
#     return XT

# # def reduce_var(input_tensor, axis=None, keep_dims=False):
# #     mean = tf.reduce_mean(input_tensor, axis=axis, keep_dims=True)
# #     square_of_diff = tf.square(input_tensor - mean)
# #     return tf.reduce_mean(square_of_diff, axis=axis, keep_dims=keep_dims)
# # 
# # def reduce_std(input_tensor, axis=None, keepdims=False):
# #     return tf.sqrt(reduce_var(input_tensor, axis=axis, keep_dims=keepdims))
# # 
# # def reduce_cov(input_tensors):
# #     input_tensor = tf.concat(input_tensors, axis=0)
# #     mean = tf.reduce_mean(input_tensor, axis=1, keep_dims=True)
# #     ExEy = tf.matmul(mean, tf.transpose(mean))
# #     Exy  = tf.matmul(input_tensor, tf.transpose(input_tensor))/tf.cast(tf.shape(input_tensor)[1], tf.float64)
# #     return Exy-ExEy
# # 
# # Marginal Probability Density Function
# # def reduce_mpdf(input_tensors):
# #     input_tensor = tf.concat(input_tensors, axis=0)
# #     coef = tf.div(tf.pow(1/np.sqrt(2*np.pi),tf.shape(input_tensor)[0]/2) , tf.sqrt(tf.linalg.det(reduce_cov(input_tensor))))
# #     mean = tf.reduce_mean(input_tensor, axis=1, keep_dims=True)
# #     diff = input_tensor - mean
# #     power = -tf.matmul(tf.matmul(tf.transpose(diff),tf.linalg.inv(reduce_cov(input_tensor))),diff)/2
# #     mpdf = coef*tf.linalg.expm(power)
# #     return mpdf
# # 
# # def mutualinfo(X, Y):
# #     return 0

# # Placeholders for data
# X0 = tf.placeholder(tf.float64, [2, None])
# Z = tf.placeholder(tf.float64, [2, None])

# # Generate some training data
# X0_train = np.random.rand(2,train_n)
# Z_train = np.dot(R, X0_train) + noise_amp * np.random.randn(2,train_n)

# # Generate some training data
# X0_test = np.random.rand(2,test_n)
# Z_test = np.dot(R, X0_test) + noise_amp * np.random.randn(2,test_n)

# def batch_feeder():
#     for k in range(N_batch-1):
#         yield X0_train[:,k*batch_size:(k+1)*batch_size], Z_train[:,k*batch_size:(k+1)*batch_size]
#     yield X0_train[:,(N_batch-1)*batch_size:], Z_train[:,(N_batch-1)*batch_size:]


# # Change Model parameters to tensorflow constant
# R_ = tf.constant(R)
# l_ = tf.cast(tf.constant(l),tf.float64)

# # just some starting value, could be random as well
# A = {'A0':tf.Variable(np.random.rand(2,2)),
#       'A1':tf.Variable(np.random.rand(2,2)),
#       'A2':tf.Variable(np.random.rand(2,2))
#       }
# T = len(A)
# T_ = tf.constant(T, dtype=tf.float64)

# A0 = tf.Variable(np.random.rand(2,2))

# # Model definition
# # XT, (X1, X2) = forward(A,X0)
# XT = A0_forward(A0, X0, T_)

# # Optimizer
# cost = rmse(Z, XT) + regularization(l_, A)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) 

# # Mutual information
# # mpdf = reduce_mpdf(X0)
# # I_X1X0 = mutualinfo(X1,X0)
# # I_X2X0 = mutualinfo(X2,X0)
# # I_XTX0 = mutualinfo(XT,X0)
# # I_X1Z = mutualinfo(X1,Z)
# # I_X2Z = mutualinfo(X2,Z)
# # I_XTZ = mutualinfo(XT,Z)

# # std = reduce_std(X0, axis=1, keepdims=True)
# # cov = reduce_cov([X0, Z])
# # det = reduce_det(cov)

# train_cost_history_A0 = np.zeros(N_iteration)
# test_cost_history_A0 = np.zeros(N_iteration)

# # First, we need to create a Tensorflow session object

# start = time.time()
# with tf.Session() as sess:
#     # Initialize all defined variables
#     tf.global_variables_initializer().run()

#     init_time = time.time()-start
#     for i in range(N_epoch):
#         for k, (X0_train_feeder, Z_train_feeder) in enumerate(batch_feeder()):
#             _, train_cost = sess.run(fetches=[optimizer, cost], feed_dict={X0: X0_train_feeder, Z: Z_train_feeder})
#             # curr_mpdf = sess.run(fetches=mpdf, feed_dict={X0: X0_train, Z: Z_train})
#             # train_cost, curr_std, curr_cov, curr_det = sess.run(fetches=[cost, std, cov, det], feed_dict={X0: X0_train, Z: Z_train})
#             test_cost = sess.run(fetches=cost, feed_dict={X0: X0_test, Z: Z_test})
#             train_cost_history_A0[i*N_batch+k] = train_cost
#             test_cost_history_A0[i*N_batch+k] = test_cost

#             # print(curr_mpdf)
#             # print(curr_cov,"\n", np.cov(np.concatenate((X0_train, Z_train), axis=0)))
#             # print(curr_det,"\n", np.linalg.det(np.cov(np.concatenate((X0_train, Z_train), axis=0))))
#     duration = time.time() - start
#     final_A0 = sess.run(A0)
#     final_A0_transpose = np.transpose(final_A0)
#     R_hat_A0 = np.matmul(expm(T*(final_A0-final_A0_transpose)),expm(T*final_A0_transpose))
#     print("R_hat = %s" % R_hat_A0)
#     print("R = %s" % R)

# init_A0 = init_time
# duation_A0 = duration - init_A0

# XT, (X1, X2) = forward(A,X0)
# cost = rmse(Z, XT)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) 

# train_cost_history_At = np.zeros(N_iteration)
# test_cost_history_At = np.zeros(N_iteration)

# start = time.time()
# with tf.Session() as sess:
    
#     # Initialize all defined variables
#     tf.global_variables_initializer().run()

#     init_time = time.time()-start
#     for i in range(N_epoch):
#         for k, (X0_train_feeder, Z_train_feeder) in enumerate(batch_feeder()):
#             _, train_cost = sess.run(fetches=[optimizer, cost], feed_dict={X0: X0_train_feeder, Z: Z_train_feeder})
#             # curr_mpdf = sess.run(fetches=mpdf, feed_dict={X0: X0_train, Z: Z_train})
#             # train_cost, curr_std, curr_cov, curr_det = sess.run(fetches=[cost, std, cov, det], feed_dict={X0: X0_train, Z: Z_train})
#             test_cost = sess.run(fetches=cost, feed_dict={X0: X0_test, Z: Z_test})
#             train_cost_history_At[i*N_batch+k] = train_cost
#             test_cost_history_At[i*N_batch+k] = test_cost

#             # print(curr_mpdf)
#             # print(curr_cov,"\n", np.cov(np.concatenate((X0_train, Z_train), axis=0)))
#             # print(curr_det,"\n", np.linalg.det(np.cov(np.concatenate((X0_train, Z_train), axis=0))))
    
#     duration = time.time() - start
#     final_A = sess.run(A)
#     R_hat_At = np.identity(dim_n)
#     for key, Ai in final_A.items():
#         print("%s = %s" % (key, Ai))
#         # R_hat = np.dot(np.add(I,Ai),R_hat)
#         R_hat_At = np.dot(expm(Ai),R_hat_At)
#     print("R_hat = %s" % R_hat_At)
#     print("R = %s" % R)

# init_At = init_time
# duation_At = duration - init_At

# import matplotlib.pyplot as plt
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

# fontsize = 20
# fig, ax = plt.subplots(1,1, figsize=(9,7))
# # ax.semilogy(train_cost_history_A0, label=r'train $\phi_{T;0}$')
# # ax.semilogy(test_cost_history_A0, label=r'test $\phi_{T;0}$')
# ax.semilogy(train_cost_history_At, label=r'train $\prod e^{A_t}$')
# ax.semilogy(test_cost_history_At, label=r'test $\prod e^{A_t}$')
# ax.legend(fontsize=fontsize-5)
# ax.tick_params(labelsize=fontsize)
# ax.set_xticks(np.arange(0,N_iteration+N_batch,N_batch))
# ax.set_ylim([0.01,5.1])
# ax.set_xlabel('Iteration', fontsize=fontsize)
# ax.set_ylabel('Cost', fontsize=fontsize)
# ax.set_title('Linear Cont.-Time Neural Network', fontsize=fontsize+2)
# plt.text(N_iteration*0.3,1.8,r'$J=E\left[\ \frac{1}{2}|X_T-Z|^2\right]$', fontsize=fontsize)
# plt.text(N_iteration*0.2,0.6,r'$R=$',fontsize=fontsize)
# plt.text(N_iteration*0.3,0.5,' {}  {}\n {}  {}'.format(R[0,0],R[0,1],R[1,0],R[1,1]), fontsize=fontsize)
# # plt.text(N_iteration*0.5,0.6,r'$\hat R_{A_0}=$', fontsize=fontsize)
# # plt.text(N_iteration*0.65,0.5,' {:.2f}  {:.2f}\n {:.2f}  {:.2f}'.format(R_hat_A0[0,0],R_hat_A0[0,1],R_hat_A0[1,0],R_hat_A0[1,1]), fontsize=fontsize)
# plt.text(N_iteration*0.5,0.25,r'$\hat R_{A_t}=$', fontsize=fontsize)
# plt.text(N_iteration*0.65,0.2,' {:.2f}  {:.2f}\n {:.2f}  {:.2f}'.format(R_hat_At[0,0],R_hat_At[0,1],R_hat_At[1,0],R_hat_At[1,1]), fontsize=fontsize)
# # plt.text(N_iteration*0.3,0.08,'exec. time for $A_0$: {:.2f}+{:.2f} sec.,\n exec. time for $A_t$: {:.2f}+{:.2f} sec.'.format(init_A0, duation_A0, init_At, duation_At), fontsize=fontsize)
# plt.show()