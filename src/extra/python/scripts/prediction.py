import numpy as np
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.layers import LeakyReLU
#from keras.optimizers import SGD
from keras.optimizers import Adam
#from keras.metrics import RootMeanSquaredError
#from keras.metrics import CategoricalCrossentropy
from keras.layers import Dropout
from keras.constraints import maxnorm
from keras.models import model_from_json
#from sklearn.metrics import mean_squared_error
import math
import tensorflow as tf
#import matplotlib.pyplot as plt
#import Big_array as BA
#import cartopy

def normal(array):
    normal_values = np.loadtxt('sigma_min_max_9_day.csv', delimiter = ',', dtype = float)

    for i in range(82):
        array[i] = (array[i,:] - normal_values[i,0])/(normal_values[i,1]-normal_values[i,0])
    
    return array

def denormal(array, T_or_q):
    normal_values = np.loadtxt('sigma_min_max_9_day.csv', delimiter = ',', dtype = float)
    
    if T_or_q == 'T':
        for i in range(40):
            array[i,:] = array[i,:]*(normal_values[84+i,1]-normal_values[84+i,0]) + normal_values[84+i,0]

    elif T_or_q == 'q':
        for i in range(40):
            array[i,:] = array[i,:]*(normal_values[124+i,1]-normal_values[124+i,0]) + normal_values[124+i,0]
    return array

'''
    
def predict(res):
    # reads in model
    json_file=open('model_1_7_512.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)

    model.load_weights('model_1_7_512weights.h5')
    model.summary()
    X = normal(res)
    X = np.transpose(X)
    X1 = X[:,:]
    print(X1)
    print(X.shape)

    Y = model.predict(X1)
    print(Y)
    output = denormal(Y, 'T')

    return output

'''



def predict(data, model_name, T_or_q):
# loads an ANN model
    json_file=open(f'{model_name}.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)

    model.load_weights(f'{model_name}weights.h5')

    # Gets our array in and makes it look the same as it does for training
    #Big_Array = np.loadtxt("sigma_unseen.csv", delimiter = ",", dtype = float)
    Big_Array = normal(data)
    mini = np.amin(Big_Array)
    maxi = np.amax(Big_Array)
    print(f'The min is {mini} and the max is {maxi}')
    Big_Array = np.transpose(Big_Array)

    x = Big_Array[:,:84]
    '''
    y = Big_Array[:,214:284]


    nt            = x.shape[0]
    n_train       = int(0.8*nt)

    trainx, testx = x[:n_train, :], x[n_train:, :]
    trainy, testy = y[:n_train], y[n_train:]
    '''
    # X is what we plug into the NN
    # X = testx[:,:]
    #X = x[:,:]

    # Y is the output from our NN with the X values plugged in
    Y = model.predict(x)
    #print(Y)

    output = np.transpose(Y)
    output = denormal(output, T_or_q)
    
    mini = np.amin(output)
    maxi = np.amax(output)
    print(f'The min is {mini} and the max is {maxi}')
    return output

def scatter_plot(region, NN_name, T_or_q):
    BA = np.loadtxt('sigma_unseen.csv', delimiter = ',', dtype = float)
    pred = predict(NN_name, T_or_q)
    true_array = np.zeros((40, 96))
    pred_array = np.zeros((40, 96))
    if T_or_q == 'T':
        m = 84
    elif T_or_q == 'q':
        m = 124
    for i in range(40):
        for j in range(24):
            for k in range(4):
                true_array[i,(4*j)+k] = BA[m+i,(320*j)+(region*4)+k-1]
                pred_array[i,(4*j)+k] = pred[i,(320*j)+(region*4)+k-1]
    #return pred_array
    if np.amax(true_array) > np.amax(pred_array):
        max_val = np.amax(true_array)
    elif np.amax(true_array) == np.amax(pred_array):
        max_val = np.amax(true_array)
    else:
        max_val = np.amax(pred_array)
    plt.figure()
    plt.xlim(0,max_val)
    plt.ylim(0,max_val)
    plt.xlabel('true std value')
    plt.ylabel('ANN guess std value')
    for i in range(96):
        plt.scatter(true_array[:,i], pred_array[:,i])
    plt.show()

def best_region(NN_name, T_or_q):
    rmsem1 = 10
    p_val = 0
    for p in range(80):
        BA = np.loadtxt('sigma_unseen.csv', delimiter = ',', dtype = float)
        pred = predict(NN_name, T_or_q)
        true_array = np.zeros((40, 96))
        pred_array = np.zeros((40, 96))
        if T_or_q == 'T':
            m =84
        elif T_or_q == 'q':
            m = 124
        for i in range(40):
            for j in range(24):
                for k in range(4):
                    true_array[i,(4*j)+k] = BA[m+i,(320*j)+(p*4)+k-1]
                    pred_array[i,(4*j)+k] = pred[i,(320*j)+(p*4)+k-1]

        mse = mean_squared_error(true_array, pred_array)
        rmse = math.sqrt(mse)
        if rmse < rmsem1:
            rmsem1 = rmse
            p_val = p

        else:
            pass
    print(f'The best RMSE is for for region {p_val} and is {rmsem1}')

def worst_region(NN_name, T_or_q):
    rmsem1 = 0
    p_val = 0
    for p in range(80):
        BA = np.loadtxt('sigma_unseen.csv', delimiter = ',', dtype = float)
        pred = predict(NN_name, T_or_q)
        true_array = np.zeros((40, 96))
        pred_array = np.zeros((40, 96))
        if T_or_q == 'T':
            m = 84
        elif T_or_q == 'q':
            m = 124
        for i in range(40):
            for j in range(24):
                for k in range(4):
                    true_array[i,(4*j)+k] = BA[m+i,(320*j)+(p*4)+k-1]
                    pred_array[i,(4*j)+k] = pred[i,(320*j)+(p*4)+k-1]

        mse = mean_squared_error(true_array, pred_array)
        rmse = math.sqrt(mse)
        if rmse > rmsem1:
            rmsem1 = rmse
            p_val = p

        else:
            pass
    print(f'The worst RMSE is for for region {p_val} and is {rmsem1}')


def region_RMSE(NN_name, T_or_q, region_num):

    BA = np.loadtxt('sigma_unseen.csv', delimiter = ',', dtype = float)
    pred = predict(BA, NN_name, T_or_q)
    true_array = np.zeros((40, 96))
    pred_array = np.zeros((40, 96))
    if T_or_q == 'T':
        m = 84
    elif T_or_q == 'q':
        m = 124
    for i in range(40):
        for j in range(24):
            for k in range(4):
                true_array[i,(4*j)+k] = BA[m+i,(320*j)+(region_num*4)+k-1]
                pred_array[i,(4*j)+k] = pred[i,(320*j)+(region_num*4)+k-1]
    mse = mean_squared_error(true_array, pred_array)
    rmse = math.sqrt(mse)
    return rmse
            
def RMSE_val():
    RMSEs = np.zeros((80,2))
    for i in range(80):
        RMSEs[i,0] = i + 1
        RMSEs[i,1] = region_RMSE('New_network', 'T', i) 
    np.savetxt('RMSE_vals_q.csv', RMSEs, delimiter = ',')
'''
for i in range(100):
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('true std value')
    plt.ylabel('ANN guess std value')
    plt.scatter(testy[i,:], Y[i,:])
plt.show()
'''

# some early ideas for graphs comparing real STD and neural network prediction STD
levels = np.arange(40)
def MLSTD(col, original, predict):
    plt.figure()
    plt.plot(original[:,col],levels, label = f'{col} true')
    plt.plot(predict[:,col],levels,  label = f'{col} predict')
    plt.xlabel('standard deviation of temperature')
    plt.ylabel('model level')
    plt.title('comparison of true and predicted standard deviations of temperature')
    plt.legend()
    plt.show()

#for i in range(10):
#    MLSTD(i)
