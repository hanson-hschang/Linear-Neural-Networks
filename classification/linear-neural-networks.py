import sys
import os
import tensorflow as tf
import numpy as np
from scipy.linalg import expm
import time
from data import read_data, train_test_split, batch_feeder
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization! 

class Model:
    def __init__(self, data_info, hyper_param):

        self.data_info = data_info
        self.hyper_param = hyper_param

        # Placeholders for data
        self.X0 = tf.placeholder(tf.float64, [data_info['input_data_dimension'], None])
        self.Z = tf.placeholder(tf.float64, [data_info['output_data_dimension'], None])

        # Model definition
        self.A = dict()
        for i in range(hyper_param['inner_layers']+1):
            self.A['A'+str(i)] = tf.Variable(np.random.rand(data_info['input_data_dimension'],data_info['input_data_dimension']))
        if hyper_param['model']=='forward':   
            self.XT, self.Xi = self.forward()
        elif hyper_param['model']=='forward_exp':   
            self.XT, self.Xi = self.forward_exp()
        self.beta = tf.constant(hyper_param['beta'])
        self.output =  tf.sigmoid(tf.matmul(self.beta,self.XT))
        self.predeiction = self.output / tf.reduce_sum(self.output,0)

        # Cost Function
        self.cost = self.log_lost() + self.regularization()

        # Optimizer
        self.optimizer = tf.train.GradientDescentOptimizer(self.hyper_param['learning_rate']).minimize(self.cost)

        self.accuracy_percentage = self.accuracy()*100

        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()
    
    # Forward Network
    def forward(self):
        """Forward pass for our fuction"""
        Xi = dict(X0=self.X0)
        for i in range(hyper_param['inner_layers']+1):
            Xi['X'+str(i+1)] = tf.matmul(tf.constant(np.identity(self.data_info['input_data_dimension']))+self.hyper_param['dt']*self.A['A'+str(i)], \
                                            Xi['X'+str(i)])
        return Xi['X'+str(hyper_param['inner_layers']+1)], Xi

    # Forward Network with Exponetial Weight
    def forward_exp(self):
        """Forward pass for our fuction"""
        Xi = dict(X0=self.X0)
        for i in range(hyper_param['inner_layers']+1):
            Xi['X'+str(i+1)] = tf.matmul(tf.linalg.expm(self.hyper_param['dt']*self.A['A'+str(i)]),Xi['X'+str(i)])
        return Xi['X'+str(hyper_param['inner_layers']+1)], Xi
    
    # Cross Entropy Lost Function
    def log_lost(self):
        """Compute log lost"""
        return -tf.reduce_mean(self.Z*tf.log(self.predeiction))

    # Regularized Lost Function
    def regularization(self):
        """Compute regularization cost"""
        trace_int = 0
        for i in range(hyper_param['inner_layers']+1):
            trace_int = trace_int + tf.reduce_sum(tf.square(self.A['A'+str(i)]) * self.hyper_param['dt'])
        return self.hyper_param['lambda']*trace_int

    # Accuracy 
    def accuracy(self):
        prediction = tf.argmax(self.output, 0)
        label = tf.argmax(self.Z, 0)
        equality = tf.equal(prediction,label)
        return tf.reduce_mean(tf.cast(equality, tf.float64))

    # Start Training
    def train(self, X0_train, Z_train, X0_validate, Z_validate, X0_test, Z_test):
        train_cost_history = np.zeros(self.hyper_param['N_iteration'])
        validate_accuracy_history = np.zeros(self.hyper_param['N_iteration'])

        # First, we need to create a Tensorflow session object
        with tf.Session() as sess:
            # Initialize all defined variables
            tf.global_variables_initializer().run()

            for i in range(self.hyper_param['N_epoch']):
                for k, (X0_train_feeder, Z_train_feeder) in enumerate(batch_feeder(X0_train, Z_train, batch_size=self.hyper_param['batch_size'])):
                    _, train_cost = sess.run(fetches=[self.optimizer, self.cost], feed_dict={self.X0: X0_train_feeder, self.Z: Z_train_feeder})
                    validate_accuracy = sess.run(fetches=self.accuracy_percentage, feed_dict={self.X0: X0_validate, self.Z: Z_validate})
                    train_cost_history[i*N_batch+k] = train_cost
                    validate_accuracy_history[i*N_batch+k] = validate_accuracy
            
            self.save_path = self.saver.save(sess, os.getcwd()+"/model.ckpt")
            print("Model saved in path: %s" % self.save_path)

            final_A_dict = sess.run(self.A)
            final_XT = sess.run(fetches=self.XT, feed_dict={self.X0: X0_train, self.Z: Z_train})

            print("Test Accuracy:", sess.run(fetches=self.accuracy_percentage, feed_dict={self.X0: X0_test, self.Z: Z_test}), "%")

        # Change Weight A from Dictionary to List
        def convert_dict_to_list(A_dict):
            A_list = [None] * len(A_dict)
            for i in range(len(A_dict)):
                A_list[i] = A_dict['A'+str(i)]
            return A_list
        
        # Estimate model vector from Weight A
        def estimate_w(A_dict, model, dt):
            A_list = convert_dict_to_list(A_dict)
            dim_n = A_list[0].shape[0]
            normal_to_w_hat = np.zeros([dim_n,dim_n-1])
            normal_to_w_hat[0,:] = np.ones(dim_n-1)
            normal_to_w_hat[1,:] = np.ones(dim_n-1)
            if dim_n>2:
                for j in range(dim_n-2):
                    normal_to_w_hat[j+2,j] = 1
            N = len(A_list)
            if model == 'forward':
                I = np.identity(dim_n)
                for j in range(dim_n-1):
                    for i in range(N):
                        normal_to_w_hat[:,j] = np.dot(np.linalg.inv(I+dt*A_list[N-i-1]), normal_to_w_hat[:,j])
                    normal_to_w_hat[:,j] = normal_to_w_hat[:,j]/np.linalg.norm(normal_to_w_hat[:,j])
            elif model == 'forward_exp':
                for j in range(dim_n-1):
                    for i in range(N):
                        normal_to_w_hat[:,j] = np.dot(np.linalg.inv(expm(dt*A_list[N-i-1])), normal_to_w_hat[:,j])
                    normal_to_w_hat[:,j] = normal_to_w_hat[:,j] / np.linalg.norm(normal_to_w_hat[:,j])
            
            def find_hyperplane_vector(vectors):
                dim_n = vectors.shape[0]
                extend_vectors = np.zeros([dim_n-1,2*dim_n])
                extend_vectors[:,:dim_n] = np.transpose(vectors)
                extend_vectors[:,dim_n:] = np.transpose(vectors)
                
                normal_vector = np.zeros(dim_n)
                for i in range(dim_n):
                    normal_vector[i] = ((-1.)**(i*(dim_n+1))) * np.linalg.det(extend_vectors[:,i+1:i+dim_n])
                normal_vector = ((-1.)**(dim_n+1)) * normal_vector / np.linalg.norm(normal_vector)
                
                return normal_vector
            
            w_hat = find_hyperplane_vector(normal_to_w_hat)

            return w_hat, normal_to_w_hat 
        
        [w_hat, normal_to_w_hat] = estimate_w(final_A_dict, self.hyper_param['model'], hyper_param['dt'])
            
        return train_cost_history, validate_accuracy_history, w_hat, normal_to_w_hat, final_XT

    def test(self, X0_test, Z_test):
        test_size = X0_test.shape[1]
        test_accuracy_history = np.zeros(test_size)
        with tf.Session() as sess:
            # Restore variables from disk.
            self.saver.restore(sess, self.save_path)
            print("Model restored.")
            tf.global_variables_initializer().run()

            
        return

# Split data into train, validate and test dataset
def read_and_split_data_from_file(file_name):
    data_X0, data_Z, data_info = read_data(file_name)
    data_info['train_size'], data_info['validate_size'], data_info['test_size'] = 0.7, 0.2, 0.1
    X0_train, Z_train, X0_validate, Z_validate, X0_test, Z_test = \
        train_test_split(X=data_X0, Z=data_Z, \
                            train_size=data_info['train_size'], \
                            validate_size=data_info['validate_size'])

    return X0_train, Z_train, X0_validate, Z_validate, X0_test, Z_test, data_info

# Plot the cost graph
def plot_cost(cost):
    # plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    fontsize = 20
    fig, ax = plt.subplots(1,1, figsize=(9,7))
    ax.semilogy(cost, label=r'train')
    ax.legend(fontsize=fontsize-5)
    ax.tick_params(labelsize=fontsize)
    # ax.set_ylim([0.01,5.1])
    ax.set_xlabel('Iteration', fontsize=fontsize)
    ax.set_ylabel('Cost', fontsize=fontsize)
    ax.set_title('Linear Cont.-Time Neural Network', fontsize=fontsize+2)
    return 

# Plot the accuracy graph
def plot_accuracy(accuracy):
    # plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    fontsize = 20
    fig, ax = plt.subplots(1,1, figsize=(9,7))
    ax.plot(accuracy)
    # ax.legend(fontsize=fontsize-5)
    ax.tick_params(labelsize=fontsize)
    # ax.set_ylim([0.01,5.1])
    ax.set_xlabel('Iteration', fontsize=fontsize)
    ax.set_ylabel('Accuracy', fontsize=fontsize)
    ax.set_title('Linear Cont.-Time Neural Network', fontsize=fontsize+2)
    return 

# Plot the data cloud point with estimate vector
def plot_model_with_data(X0, Z, w_hat, normal_to_w_hat):
    # plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    fontsize = 20
    if w_hat.shape[0]==2 :    
        fig, ax = plt.subplots(1,1, figsize=(9,7))
        ax.scatter(X0[0,:],X0[1,:],c=Z[0,:])
        ax.arrow(0, 0, w_hat[0], w_hat[1], head_width=0.05, head_length=0.1, fc='r', ec='r')
        ax.arrow(0, 0, normal_to_w_hat[0,0], normal_to_w_hat[1,0], head_width=0.05, head_length=0.1, fc='k', ec='k')
        plt.axis('equal')
    elif w_hat.shape[0]==3 :
        fig, ax = plt.subplots(1,1, figsize=(9,7))
        ax = plt.axes(projection='3d')
        ax.scatter(X0[0,:],X0[1,:],X0[2,:],c=Z[0,:])
        xline = [0, w_hat[0]]
        yline = [0, w_hat[1]]
        zline = [0, w_hat[2]]
        ax.plot3D(xline, yline, zline, 'r')

        xline = [0, normal_to_w_hat[0,0]]
        yline = [0, normal_to_w_hat[1,0]]
        zline = [0, normal_to_w_hat[2,0]]
        ax.plot3D(xline, yline, zline, 'k')

        xline = [0, normal_to_w_hat[0,1]]
        yline = [0, normal_to_w_hat[1,1]]
        zline = [0, normal_to_w_hat[2,1]]
        ax.plot3D(xline, yline, zline, 'k')
        
        plt.axis('equal')

    return

if __name__ == '__main__':
    X0_train, Z_train, \
    X0_validate, Z_validate, \
    X0_test, Z_test, \
    data_info = read_and_split_data_from_file(file_name=sys.argv[1])

    hyper_param = dict()
    hyper_param['batch_size'] = 10
    hyper_param['N_epoch'] = 10
    hyper_param['learning_rate'] = 5e-2
    N_batch = Z_train.shape[1] // hyper_param['batch_size']
    hyper_param['N_iteration'] = N_batch * hyper_param['N_epoch']

    hyper_param['beta'] = np.zeros([2,data_info['input_data_dimension']])
    hyper_param['beta'][0,0] = 1.
    hyper_param['beta'][1,1] = 1.

    hyper_param['T'] = 1.
    hyper_param['inner_layers'] = 3
    hyper_param['dt'] = hyper_param['T']/(hyper_param['inner_layers']+1.)
    hyper_param['lambda'] = 0

    if sys.platform=='linux':
        hyper_param['model'] = 'forward_exp' # for Linux version of tf
    else:
        hyper_param['model'] = 'forward' # for other version of tf 

    model_At = Model(data_info, hyper_param)
    train_cost, validate_accuracy, w_hat, normal_to_w_hat, XT = model_At.train(X0_train, Z_train, X0_validate, Z_validate, X0_test, Z_test)

    print('esitmate w:',w_hat)
    print('model w:',data_info['relation_w'])

    # plot_cost(train_cost)
    # plot_model_with_data(X0_validate, Z_validate, w_hat, normal_to_w_hat)

    # hyper_param['inner_layers'] = 0
    # hyper_param['dt'] = 1
    # model_A0 = Model(data_info, hyper_param)
    # train_cost, validate_accuracy, w_hat, normal_to_w_hat = model_A0.train(X0_train, Z_train, X0_validate, Z_validate, X0_test, Z_test)

    # print(w_hat)
    print('\n inner dot of model vector and learned vector')
    print(np.dot(w_hat,data_info['relation_w']))

    plot_cost(train_cost)
    plot_accuracy(validate_accuracy)
    plot_model_with_data(X0_validate, Z_validate, w_hat, normal_to_w_hat)
    if data_info['input_data_dimension']==2:
        plot_model_with_data(XT, Z_train, np.array([-1, 1]), np.array([[1],[1]]))
    elif data_info['input_data_dimension']==3:
        plot_model_with_data(XT, Z_train, np.array([-1, 1, 0]), np.array([[1, 1],[1, 1], [0, 1]]))


    plt.show()