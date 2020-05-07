# -*- coding: utf-8 -*-

# ======================================================================================
# preparing db for DL approach, slicing every signal in sequence of 15 consecutive peaks
# ======================================================================================


import numpy as np
from operator import itemgetter
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.layers as L
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adadelta, Nadam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
import pickle


# %%
#
# load db
# VERY IMPORTANT:
# to train CNNs use "train_idx" in line 51 of "prepare_db_for_DL.py"
# to check CNNs pre-trained performances use "test_idx" in line 51 of "prepare_db_for_DL.py"
with open('cleaned_db_for_DL_15p.pickle', 'rb') as file:
    d = pickle.load(file)



# %%

# last preparation for data 

data = [d["samples"][x] for x in range(len(d["labels"])) if d["labels"][x].quality < 0.01]

targets = [d["labels"][x] for x in range(len(d["labels"])) if d["labels"][x].quality < 0.01]

targets = list(map(itemgetter(0), targets))  # 0 == age
targets = np.asarray(targets, dtype=int)
targets = np.where(targets < 50, 1, 0)


# to normalize and positivize data
data = list(map(lambda arr: (np.asarray(arr) - np.min(arr))/np.max(np.asarray(arr) - np.min(arr)), data))
data = np.asarray(data)

t_size = 4383
v_size = 2383



# %%


# creating train, validation, test, sets

x_train, x_test, y_train, y_test = train_test_split(data, targets,
                                                    test_size=0.33,  # 0.322303
                                                    stratify=targets,
                                                    random_state=42)



x_val, x_test, y_val, y_test = train_test_split(x_test, y_test,
                                                test_size=0.33,  # 0.322303
                                                stratify=y_test,
                                                random_state=42)


x_train = np.expand_dims(x_train, 2)
x_test = np.expand_dims(x_test, 2)
x_val = np.expand_dims(x_val, 2)
data = np.expand_dims(data, 2)

# Convert class vectors to binary class matrices.
y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)
y_val = to_categorical(y_val, 2)
targets = to_categorical(targets, 2)


print("Best dummy classifier accuracy: ",
      max(np.sum(y_train, axis=0))/len(y_train))  # 0.50477  ///  # 0.974

print("Best dummy classifier test accuracy: ",
      max(np.sum(y_test, axis=0))/len(y_test))    # 0.504825 ///  # 0.974

print("Best dummy classifier validation accuracy: ",
      max(np.sum(y_val, axis=0))/len(y_val))      # 0.505    ///  # 0.97


class_w = {0:1/np.sum(y_train, axis=0)[0],
           1:1/np.sum(y_train, axis=0)[1]}


# %%

# RUN THIS CELL TO CREATE VARIABLES "inp" and "outp", needed to build "m" in cell 7

filters = 1                                # 1
num_layers = 12                            # 12
learningRate = .002                        # .002
batch_dim = 32                             # 32
dilation = 1                               # 1
kernel_dim = 50                            # 50
k_init = "lecun_uniform"                   # "lecun_uniform"
b_init = "zeros"  # constant(-1.)          # "zeros"
k_reg = l1_l2(l1=0., l2=0.)                #
b_reg = l1_l2(l1=0., l2=0.)                #
act = "elu"                                # "elu"
pad = "same"                               # "same"
opt = Nadam(learning_rate=learningRate)    # Nadam
        #    beta_1=0.99,
        #    beta_2=0.9999)
pat = 8                                   # 8


def schedule(epoch, lr):
  if epoch < 250 and lr < 1e-4:
    return .001  # possible to change this
  else:
    return lr


scheduler = LearningRateScheduler(schedule)

reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience = pat,
                            min_lr=1e-5, verbose=1, cooldown=2)


inp = L.Input(shape=x_train.shape[1:])

h = L.Conv1D(filters, kernel_dim, kernel_initializer=k_init, padding=pad,
             activation=act, dilation_rate=1,
             bias_initializer=b_init, kernel_regularizer=k_reg,
             bias_regularizer=b_reg)(inp)


outlist = [inp, h]

for i in range(1, num_layers):
  h = L.Conv1D(filters, kernel_dim, kernel_initializer=k_init, padding=pad,
               activation=act, dilation_rate=1,  # dilation**i,
               bias_initializer=b_init, kernel_regularizer=k_reg,
               bias_regularizer=b_reg)(L.add(outlist))

  outlist = [*outlist, h]

  h = L.Dropout(0.2)(h)


pre_out = L.Flatten()(L.add(outlist))

outp = L.Dense(2, activation='softmax')(pre_out)

model = Model(inp, outp)

METRICS = ['AUC', 'accuracy']


model.compile(loss='categorical_crossentropy', optimizer=opt,
              metrics=METRICS)
# checkpoint
filepath="RESULTS/weights.best-drop-{}-{}-.hdf5".format(num_layers, learningRate)

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min')

callbacks_list = [checkpoint, reducer, scheduler]
# %%

# SKIP THIS CELL IF YOU DON'T WANT TO TRAIN YOUR OWN CNN

out = model.fit(x=x_train,
                y=y_train,
                validation_data=[x_test, y_test],
                epochs=300,
                batch_size=batch_dim,
#                class_weight=class_w,
                callbacks=callbacks_list,
                shuffle=True,
                verbose=1)







# %%

# EVALUATION OF PRE-TRAINED MODEL's performances

m = Model(inp, outp)

m.compile(loss='categorical_crossentropy', optimizer=opt,
          metrics=['AUC', 'cosine_proximity', 'categorical_hinge', 'hinge',
                   'FalsePositives', 'FalseNegatives', 'TruePositives',
                   'TrueNegatives', 'Precision', 'Recall', 'accuracy',
                   'Poisson', 'LogCoshError', 'mse', 'KLDivergence'])

m.load_weights(filepath=filepath)



# Re-evaluate the model
m.evaluate(data, targets, verbose=2)


y_predicted = m.predict(data)
