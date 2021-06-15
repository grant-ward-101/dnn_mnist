import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('mnist_train.csv')
test_data = pd.read_csv('mnist_test.csv')

train_x = train_data.iloc[:, 1:].to_numpy()
train_y = train_data.iloc[:, 0].to_numpy()

test_x = test_data.iloc[:, 1:].to_numpy()
test_y = test_data.iloc[:, 0].to_numpy()

# normalize input set
# convert data from (0, 255) to (0, 1)
train_x = train_x / 255
test_x = test_x / 255

# reshape input set
train_x = train_x.reshape(-1, 784)
test_x = test_x.reshape(-1, 784)

# split train set and validation set
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.1, random_state=2)

# convert label set
# from number to list of 10 elements for 10 number, the element of number is 1, others are 0
# label 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
# label 5 -> [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
train_y = to_categorical(train_y, num_classes=10)
test_y = to_categorical(test_y, num_classes=10)
val_y = to_categorical(val_y, num_classes=10)
# model architecture
# 3 layers
# each layer uses an activation function and an dropout rate for overfitting
model = Sequential()
# first layer and second layer
# use 256 hidden unit
# use dropout rate as 0.45
# use relu activation function
model.add(Dense(256, input_dim=784))
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.4))

# third layer
# use 10 units as 10 label outputs
# use softmax activation function
model.add(Dense(10))
model.add(Activation('softmax'))

# compile model
# use categorical_crossentropy as loss function
# use adam optimizer as optimization
# use accuravy as performance metric
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = np.arange(1, 6, 1)
train_acc = []
train_loss = []
val_acc = []
val_loss = []

for epoch in epochs:
    # train the compiled model
    model.fit(train_x, train_y, epochs=epoch, batch_size=32)

    loss_train, accuracy_train = model.evaluate(train_x, train_y, batch_size=32)
    train_acc.append(accuracy_train)
    train_loss.append(loss_train)
    # evaluating the model performance with validation set
    loss_val, accuracy_val = model.evaluate(val_x, val_y, batch_size=32)
    val_acc.append(accuracy_val)
    val_loss.append(loss_val)
    # evaluating the model performance with testing set
    loss_test, accuracy_test = model.evaluate(test_x, test_y, batch_size=32)

plt.plot(epochs, train_acc, label='train_acc')
plt.plot(epochs, train_loss, label='train_loss')
plt.plot(epochs, val_acc, label='val_acc')
plt.plot(epochs, val_loss, label='val_loss')
plt.legend()
plt.show()
