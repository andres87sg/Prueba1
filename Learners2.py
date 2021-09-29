# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 17:00:39 2021

@author: Andres
"""
#%%

import matplotlib.pyplot as plt

#%%

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

#%%

def create_keras_model():
    """
    This function compiles and returns a Keras model.
    Should be passed to KerasClassifier in the Keras scikit-learn API.
    """

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    return model
#%%

# create the classifier
classifier = KerasClassifier(create_keras_model)
#model = create_keras_model()

#%%

import numpy as np
from tensorflow.keras.datasets import mnist

# read training data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(10000, 28, 28, 1).astype('float32') / 255
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# assemble initial data
n_initial = 1000
initial_idx = np.random.choice(range(len(X_train)), size=n_initial, replace=False)
X_initial = X_train[initial_idx]
y_initial = y_train[initial_idx]

# generate the pool
# remove the initial data from the training dataset
X_pool = np.delete(X_train, initial_idx, axis=0)[:5000]
y_pool = np.delete(y_train, initial_idx, axis=0)[:5000]

# #%%

# history = classifier.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10)


#%%

from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from modAL.expected_error import expected_error_reduction

#%%

# initialize ActiveLearner
learner = ActiveLearner(estimator=classifier, 
                        query_strategy=uncertainty_sampling, 
                        #bootstrap_init=True,
                        X_training=X_initial,
                        y_training=y_initial,
                        verbose=1
                        )
                        



#predictions=learner.score(X_test,y_test)

#%%

performance_history = []

learner_list=[]
n_queries = 10
for idx in range(n_queries):
    print('Query no. %d' % (idx + 1))
    query_idx, query_instance = learner.query(X_pool, n_instances=50, verbose=0)
    #print(query_idx)
    learner.fit(
        X=X_pool[query_idx], y=y_pool[query_idx]
    )
    model_accuracy = learner.score(X_test, y_test)
    performance_history.append(model_accuracy)
    


#%%

predictions=learner.score(X_test,y_test)


#%%    
    #model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    #model.fit(X_pool[query_idx],y_pool[query_idx],epochs=1,verbose=2)
    #model.fit(X_pool[query_idx],y_pool[query_idx],epochs=1,verbose=1)
    #predictions=model.score(X_test,y_test)
    
    #learner = ActiveLearner(estimator=classifier, X_training=X_pool[query_idx], y_training=y_pool[query_idx],bootstrap_init=True,verbose=1)

    # remove queried instance from pool
    # X_pool = np.delete(X_pool, query_idx, axis=0)
    # y_pool = np.delete(y_pool, query_idx, axis=0)
    #learner_list.append(learner)
    
# %%
# classifier = KerasClassifier(model)

#%%

X_test = X_pool[query_idx]
y_test = y_pool[query_idx]

predictions=model.predict(X_test)
predicted = np.round(predictions)

#%%

# #%%    

# from modAL.models import ActiveLearner, Committee

# committee = Committee(learner_list)
    
# #predictions=learner.score(X_test,y_test)
    
# for learner_idx, learner in enumerate(committee):
#     learner.score(X_test,y_test)

#%%

from tensorflow.keras.utils import to_categorical

grtr = []
for i in range(len(y_test)):
    grtrA = np.argmax(y_test[i])
    grtr.append(grtrA)

grtr_array= np.array(grtr)

grtr = []
for i in range(len(predicted)):
    grtrA = np.argmax(predicted[i])
    grtr.append(grtrA)

predicted_array= np.array(grtr)


predicted_array==grtr_array

zz=np.sum(predicted_array==grtr_array)
print(zz)

#%print()

#%%
predictions=learner.score(X_test,y_test)

#%%

print(np.sum(predictions==grtr_array))

