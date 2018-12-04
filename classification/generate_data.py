import json
import numpy as np

file_name = "data_file_1.json"
file_name = "data/" + file_name

data_n = 10000
dim_n = 2
w = (np.random.rand(dim_n)-0.5)*2
w = w/np.linalg.norm(w)

def relation(X0, w):
    n = X0.shape[1]
    Z = np.ones(n)
    for i in range(n):
        if np.random.rand(1) >= 1/(1+np.exp(-10*np.dot(w,X0[:,i]))):
            Z[i]=-1
    return Z.reshape(1,n)

X0 = (np.random.rand(dim_n, data_n)-0.5)*2
Z = relation(X0, w)

data = dict()
for i in range(data_n):
    data[np.array2string(X0[:,i], separator=',')] = np.array2string(Z[:,i], separator=',')

file = dict()
file['file_name'] = file_name
file['input_data_dimension'] = dim_n
file['output_data_dimension'] = 1
file['number_of_data'] = data_n
file['retation_w'] = np.array2string(w, separator=',')
file['data'] = data


with open(file_name, "w") as write_file:
    json.dump(file, write_file)

