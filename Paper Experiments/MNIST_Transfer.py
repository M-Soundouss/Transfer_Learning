from keras.layers import Input, Dense
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.models import Model
from sklearn.metrics import accuracy_score
import random
import numpy as np
from collections import defaultdict
import keras.backend as K
np.random.seed(1427)

#Source Binary Classes Algorithm
def binary_source(num):
    arr = np.array(range(num))
    random.shuffle(arr)
    return arr

#Target Binary Classes Algorithm
def binary_target(arr, num, gamma):
    classes_zero = []
    classes_one = []
    for i in arr[0:int(num/2)]:
        classes_zero.append(i)
    for i in arr[int(num/2):num]:
        classes_one.append(i)
    classes_zero=np.array(classes_zero)
    classes_one=np.array(classes_one)
    random.shuffle(classes_zero)
    random.shuffle(classes_one)
    classes_target = []
    for i in classes_zero[0:int((num/2)*gamma)]:
        classes_target.append(i)
    for i in classes_one[0:int((num/2)*gamma)]:
        classes_target.append(i)
    return np.array(classes_target)

#Source Binary Outputs Algorithm
def binary_source_classify(dataset, arr, num):
    binary = []
    length = len(dataset)
    for i in range(length):
        if dataset[i] in arr[0:int(num/2)]:
            binary.append(0)
        elif dataset[i] in arr[int(num/2):num]:
            binary.append(1)
    return np.array(binary)

#Target Binary Outputs Algorithm
def binary_target_classify(dataset, arrsource, arrtarget):
    binary = []
    length = len(dataset)
    lensource = len(arrsource)
    lentarget = len(arrtarget)
    arrsource_zero = arrsource[0:int(lensource/2)]
    arrsource_one = arrsource[int(lensource/2):lensource]
    arrtarget_zero = arrtarget[0:int(lentarget / 2)]
    arrtarget_one = arrtarget[int(lentarget / 2):lentarget]
    for i in range(length):
        if dataset[i] in arrsource_zero and dataset[i] in arrtarget_zero :
            binary.append(0)
        elif dataset[i] in arrsource_one and dataset[i] in arrtarget_one :
            binary.append(1)
        elif dataset[i] not in arrtarget_zero and dataset[i] in arrsource_zero :
            binary.append(1)
        elif dataset[i] not in arrtarget_one and dataset[i] in arrsource_one :
            binary.append(0)
    return np.array(binary)

#Source MLP Model
def get_model():
    inputs = Input(shape=(784,))

    x = Dense(50, activation='relu')(inputs)
    predictions = Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_))(x)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=Adam(lr), loss='binary_crossentropy', metrics=['accuracy'])

    return model

from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

#Get data from sklearn
mnist = fetch_mldata('MNIST original')

#Split data into train and test data
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(mnist.data, mnist.target, test_size=1/7., random_state=14)

#Change pixel intensities into 0..1 range
X =X_TRAIN/255.
Y = Y_TRAIN

x_test =X_TEST/255.
y_test = Y_TEST

#Define variable values
ms = 55000
mt = 550
m = ms+mt
nbr = 10
nb_epoch = 500
batch_size = 32
lr = 0.001
n_trials = 10
l2_ = 10e-4
coef = 10e-7

#Define Source task partitions
classes_source = binary_source(nbr)

#Define Gamma dictionary
dict_gamma = {}

#For each value of Gamma, execute loop
for gamma in [0.6, 0.8, 1.0]:

    #Define performance dictionary
    dict_performance = defaultdict(list)

    #For each trial in 10 trials, execute loop
    for i in range(n_trials):

        #Define Target task partitions
        classes_target = binary_target(classes_source, nbr, gamma)
        #Split data on source and target tasks
        x_train_s = X[:ms, :]
        y_train_s = Y[:ms]

        x_train_t = X[ms:m, :]
        y_train_t = Y[ms:m]

        #Get binary classes of source and target tasks depending on the partitions
        y_train_s_binary = binary_source_classify(y_train_s, classes_source, nbr)
        y_train_t_binary = binary_target_classify(y_train_t, classes_source, classes_target)

        y_test_s_binary = binary_source_classify(y_test, classes_source, nbr)
        y_test_t_binary = binary_target_classify(y_test, classes_source, classes_target)

        #Train model on source data
        model = get_model()
        model.fit(x_train_s, y_train_s_binary, validation_split=0.1, epochs=nb_epoch, batch_size=batch_size, verbose=2)

        #Save weights for f
        model.save_weights("source_weights.h5")
        f_weights_source = model.layers[1].get_weights()[0]

        #Paper Regularization Algorithm
        def l2_reg_paper(weight_matrix, coef=coef):
            return coef*K.sum(K.square(weight_matrix-f_weights_source))

    #Fix gof
        #Use source model directly on target task
        y_pred = model.predict(x_test)
        y_pred_binary = np.array(y_pred>0.5).astype(int)

        #Save accuracy in the performance dictionary
        dict_performance['fix_gof'].append(accuracy_score(y_test_t_binary, y_pred_binary))

    #Fix f
        #Define fix f model
        def model_fix_f():
            inputs = Input(shape=(784,))
            x = Dense(50, activation='relu', trainable=False, kernel_regularizer=l2_reg_paper)(inputs) # f
            predictions = Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_))(x) # g
            model = Model(inputs=inputs, outputs=predictions)
            return model

        #Use fix f model on target task by loading saved weights of f
        model_fix_f = model_fix_f()
        model_fix_f.load_weights("source_weights.h5")
        model_fix_f.compile(optimizer=Adam(lr*100), loss='binary_crossentropy', metrics=['accuracy'])
        model_fix_f.fit(x_train_t, y_train_t_binary, validation_split=0.1, epochs=nb_epoch, batch_size=batch_size, verbose=2)
        y_pred_fix_f = model_fix_f.predict(x_test)
        y_pred_binary_fix_f = np.array(y_pred_fix_f>0.5).astype(int)

        #Save accuracy in the performance dictionary
        dict_performance['fix_f'].append(accuracy_score(y_test_t_binary, y_pred_binary_fix_f))

    #Fine tune f
        #Define fine tune f model
        def model_finetune_f():
            inputs = Input(shape=(784,))
            x = Dense(50, activation='relu', kernel_regularizer=l2_reg_paper)(inputs) # f
            predictions = Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_))(x) # g
            model = Model(inputs=inputs, outputs=predictions)
            return model

        #Use fine tune f model on target task by loading saved weights of f
        model_finetune_f = model_finetune_f()
        model_finetune_f.load_weights("source_weights.h5")
        model_finetune_f.compile(optimizer=Adam(lr*100), loss='binary_crossentropy', metrics=['accuracy'])
        model_finetune_f.fit(x_train_t, y_train_t_binary, validation_split=0.1, epochs=nb_epoch, batch_size=batch_size, verbose=2)
        y_pred_finetune_f = model_finetune_f.predict(x_test)
        y_pred_binary_finetune_f = np.array(y_pred_finetune_f>0.5).astype(int)

        #Save accuracy in the performance dictionary
        dict_performance['finetune_f'].append(accuracy_score(y_test_t_binary, y_pred_binary_finetune_f))

    #Base
        #Define base model
        def model_base():
            inputs = Input(shape=(784,))
            x = Dense(50, activation='relu')(inputs) # f
            predictions = Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_))(x) # g
            model = Model(inputs=inputs, outputs=predictions)
            model.compile(optimizer=Adam(lr*100), loss='binary_crossentropy', metrics=['accuracy'])
            return model

        # Use base model to learn T from scratch
        model_base = model_base()
        model_base.fit(x_train_t, y_train_t_binary, validation_split=0.1, epochs=nb_epoch, batch_size=batch_size, verbose=2)
        y_pred_base = model_base.predict(x_test)
        y_pred_binary_base = np.array(y_pred_base>0.5).astype(int)

        #Save accuracy in the performance dictionary
        dict_performance['base'].append(accuracy_score(y_test_t_binary, y_pred_binary_base))

    #Save accuracy values for each trial in the Gamma dictionary
    dict_gamma[gamma] = dict_performance

#Print average over trials for all Gamma values and all options
for gamma in dict_gamma:
    dict_performance = dict_gamma[gamma]
    for key, value in dict_performance.items():
        print(" %s - %s : %s "%(gamma, key, np.mean(value)))