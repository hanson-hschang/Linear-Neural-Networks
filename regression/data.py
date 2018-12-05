import sys
import json
import numpy as np


def generate_data(file_name, R, dim_n, data_n):
    file_name = "data/" + file_name

    def relation(X0, R):
        Z = np.matmul(R,X0)
        return Z

    X0 = (np.random.rand(dim_n, data_n)-0.5)*2
    Z = relation(X0, R)

    data = dict()
    for i in range(data_n):
        data[np.array2string(X0[:,i], separator=',')] = np.array2string(Z[:,i], separator=',')

    file = dict()
    file['file_name'] = file_name
    file['input_data_dimension'] = dim_n
    file['output_data_dimension'] = dim_n
    file['number_of_data'] = data_n
    file['retation_R'] = np.array2string(R, separator=',')
    file['data'] = data

    with open(file_name, "w") as write_file:
        json.dump(file, write_file)
    
    return file_name

def generate_simple_data(file_name="data_file.json", data_n=10000):
    return generate_data(file_name=file_name, R=np.array([[0., -1.],[1.,  0.]]), dim_n=2, data_n=data_n)

def read_data(file_name):
    file_name = "data/" + file_name
    with open(file_name) as f:
        file = json.load(f)
        dim_n = file["input_data_dimension"]
        data_n = file["number_of_data"]
        R = np.fromstring(file['retation_R'][1:-1], sep=',')
        data = file["data"]
        X0 = np.zeros((dim_n, data_n))
        Z = np.zeros((dim_n, data_n))

        for i in range(data_n):
            temp = data.popitem()
            X0[:,i] = np.fromstring(temp[0][1:-1], sep=',')
            Z[:,i] = np.fromstring(temp[1][1:-1], sep=',')
            
        return X0, Z


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
    sys.argv.pop(0)
    action = sys.argv.pop(0)
    if action == "generate":
        data_type = sys.argv.pop(0)
        if data_type == "simple":
            generate_simple_data()
    elif action == "read":
        file_name = sys.argv.pop(0)
        print(read_data(file_name))








