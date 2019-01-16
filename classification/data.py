import sys
import json
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def generate_data(file_name, w, noise_amp, dim_n, data_n):
    file_name = "data/" + file_name

    def relation(X0, w):
        n = X0.shape[1]
        Z = np.ones([2,n])
        for i in range(n):
            if noise_amp == 0:
                if np.dot(w,X0[:,i])>=0:
                    Z[1,i]=0
                else:
                    Z[0,i]=0
            else:
                if np.random.rand(1) >= 1/(1+np.exp(-np.dot(w,X0[:,i])/noise_amp)):
                    Z[1,i]=0
                else:
                    Z[0,i]=0
        return Z

    X0 = (np.random.rand(dim_n, data_n)-0.5)*2
    Z = relation(X0, w)

    data = dict()
    for i in range(data_n):
        data[np.array2string(X0[:,i], separator=',')] = np.array2string(Z[:,i], separator=',')

    file = dict()
    file['file_name'] = file_name
    file['input_data_dimension'] = dim_n
    file['output_data_dimension'] = 2
    file['number_of_data'] = data_n
    file['relation_w'] = np.array2string(w, separator=',')
    file['noise_amp'] = noise_amp
    file['data'] = data

    with open(file_name, "w") as write_file:
        json.dump(file, write_file)
    
    return file_name

def generate_simple_data(file_name="data_file.json", noise_amp=0.1, dim_n = 2, data_n=10000):
    w = (np.random.rand(dim_n)-0.5)*2
    w = w/np.linalg.norm(w)
    return generate_data(file_name=file_name, w=w, noise_amp=noise_amp, dim_n=dim_n, data_n=data_n)

def read_data(file_name):
    file_name = "data/" + file_name
    with open(file_name) as f:
        file_info = json.load(f)
        input_dim_n = file_info["input_data_dimension"]
        output_dim_n = file_info["output_data_dimension"]
        data_n = file_info["number_of_data"]
        file_info['relation_w'] = np.fromstring(file_info['relation_w'][1:-1], sep=',')
        data = file_info["data"]
        X0 = np.zeros((input_dim_n, data_n))
        Z = np.zeros((output_dim_n, data_n))

        for i in range(data_n):
            temp = data.popitem()
            X0[:,i] = np.fromstring(temp[0][1:-1], sep=',')
            Z[:,i] = np.fromstring(temp[1][1:-1], sep=',')
            
        return X0, Z, file_info

def plot_data(file_name):
    X0, Z, file_info = read_data(file_name)
    fig, ax = plt.subplots(1,1, figsize=(9,7))
    ax.scatter(X0[0,:],X0[1,:],c=Z[0,:])
    plt.show()
    return


def train_test_split(X, Z, train_size=0.7, validate_size=0.2):
    data_n = Z.shape[1]
    train_n = int(data_n * train_size)
    validate_n = int(data_n * validate_size)
    X_train = X[:,:train_n]
    Z_train = Z[:,:train_n]
    X_validate = X[:,train_n:train_n+validate_n]
    Z_validate = Z[:,train_n:train_n+validate_n]
    X_test = X[:,train_n+validate_n:]
    Z_test = Z[:,train_n+validate_n:]
    return X_train, Z_train, X_validate, Z_validate, X_test, Z_test

def batch_feeder(X, Z, batch_size=1):
    n = Z.shape[1]
    N_batch = n // batch_size
    for k in range(N_batch-1):
        yield X[:,k*batch_size:(k+1)*batch_size], Z[:,k*batch_size:(k+1)*batch_size]
    yield X[:,(N_batch-1)*batch_size:], Z[:,(N_batch-1)*batch_size:]


if __name__ == '__main__':
    action = sys.argv[1]
    if action == "generate":
        data_type = sys.argv[2]
        if data_type == "simple":
            dim_n = int(sys.argv[3])
            generate_simple_data(file_name="data_file_simple_{}.json".format(dim_n), dim_n=dim_n)
        elif data_type == "simple_no_noise":
            dim_n = int(sys.argv[3])
            generate_simple_data(file_name="data_file_simple_no_noise_{}.json".format(dim_n), dim_n=dim_n, noise_amp=0)
    elif action == "read":
        file_name = sys.argv[2]
        print(read_data(file_name))
    elif action == "plot":
        file_name = sys.argv[2]
        plot_data(file_name)








