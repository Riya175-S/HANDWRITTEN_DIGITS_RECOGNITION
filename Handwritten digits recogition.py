#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plot


# In[2]:


# silent all warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import warnings
warnings.filterwarnings('ignore')
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


# In[3]:


import keras
import numpy as np
import matplotlib 
from matplotlib import pyplot as plot
from keras.utils import np_utils


# In[4]:


from keras.datasets import mnist


# In[5]:


import keras
from keras import datasets
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras import backend as K
#  to split the data of training and testing sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[6]:


from keras.models import Sequential 
from keras.layers import Activation, Dense, Flatten 


# In[7]:


model = Sequential()


# In[8]:


model.add(Dense(512, input_shape=(784,)))


# In[9]:


model.add(Activation('relu'))


# In[10]:


model.add(Dropout(0.2))


# In[11]:


model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))


# In[12]:


model.add(Dense(10))


# In[13]:


model.add(Activation('softmax'))


# In[14]:


model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')


# In[15]:


(X_train, y_train), (X_test, y_test) = mnist.load_data()
print('loaded data')


# In[16]:


# printing first 10 images
for i in range(10):
    plot.subplot(3,5,i+1)
    plot.tight_layout()
    plot.imshow(X_train[i], cmap='gray', interpolation='none')
    plot.title("Digit: {}".format(y_train[i]))
    plot.xticks([])
    plot.yticks([])


# In[17]:


X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)


# In[18]:


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


# In[19]:


plot.hist(X_train[0])
plot.title("Digit: {}".format(y_train[0]))


# In[20]:


plot.hist(X_train[1])
plot.title("Digit: {}".format(y_train[1]))


# In[21]:


plot.hist(X_train[2])
plot.title("Digit: {}".format(y_train[2]))


# In[22]:


print(np.unique(y_train, return_counts=True))


# In[23]:


n_classes = 10
Y_train = np_utils.to_categorical(y_train, n_classes)


# In[24]:


for i in range(5):
   print (Y_train[i])


# In[25]:


Y_test = np_utils.to_categorical(y_test, n_classes)


# In[26]:


history = model.fit(X_train, Y_train,
   batch_size=128, epochs=20,
   verbose=2,
   validation_data=(X_test, Y_test))


# In[27]:


#To evaluate the model performance, we call evaluate method as follows 
loss_and_metrics = model.evaluate(X_test, Y_test, verbose=2)


# In[28]:



#To evaluate the model performance, we call evaluate method as follows −
loss_and_metrics = model.evaluate(X_test, Y_test, verbose=2)


# In[29]:


#We will print the loss and accuracy using the following two statements −
print("Test Loss", loss_and_metrics[0])
print("Test Accuracy", loss_and_metrics[1])


# In[30]:


plot.subplot(2,1,1)
plot.plot(history.history['accuracy'])
plot.plot(history.history['val_accuracy'])
plot.title('model accuracy')
plot.ylabel('accuracy')
plot.xlabel('epoch')
plot.legend(['train', 'test'], loc='lower right')


# In[31]:


plot.subplot(2,1,2)
plot.plot(history.history['loss'])
plot.plot(history.history['val_loss'])
plot.title('model loss')
plot.ylabel('loss')
plot.xlabel('epoch')
plot.legend(['train', 'test'], loc='upper right')


# In[32]:


predictions = (model.predict(X_test) > 0.5).astype("int32")


# In[33]:


from tensorflow import keras
print(keras.__version__)


# In[34]:


correct_predictions = np.nonzero(predictions == y_test)[0]
incorrect_predictions = np.nonzero(predictions != y_test)[0]


# In[35]:


print(len(correct_predictions)," classified correctly")
print(len(incorrect_predictions)," classified incorrectly")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




