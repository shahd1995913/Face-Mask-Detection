# import library
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
import cv2
import os
import numpy as np


"""

 Set labels for images folder, set size for img  , 
 create  a function called a get data that responsible 
 to read the folder with images and convert from  BGR to RGB
format and reshaping images to preferred size.
Args : take a string path  
return : array of float data 

"""

labels = ['mask', 'withoutmask']
img_size = 224

def get_data(data_dir):
    data = []  # create a list called data
    for label in labels: # loop in folder 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data , dtype=object)

# call the get_data with assign the path of the folder that contain the folders with images .
train = get_data('data/train')
val = get_data('data/test')

# call the get_data with assign the path of the folder that contain the folders with images .
l = [] # create a list for saved the labeled result 
for i in train:
    if(i[1] == 0):
        l.append("withoutmask")
    else:
        l.append("mask")
# sns.set_style('darkgrid')
# sns.countplot(l)

# # plot a sample image with mask 😷
# plt.figure(figsize = (5,5))
# plt.imshow(train[1][0])
# plt.title(labels[train[0][1]])

# # plot a sample image with out mask 😶
# plt.figure(figsize = (5,5))
# plt.imshow(train[-1][0])
# plt.title(labels[train[-1][1]])

# Preprocessing the dataset  by Normalization
x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in train:
  x_train.append(feature)
  y_train.append(label)

for feature, label in val:
  x_val.append(feature)
  y_val.append(label)

# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)

# use ImageDataGenerator for Generate batches of tensor image data with real-time data
datagen = ImageDataGenerator(
        featurewise_center=False, 
        samplewise_center=False, 
        featurewise_std_normalization=False,
        samplewise_std_normalization=False, 
        zca_whitening=False, 
        rotation_range = 30,  
        zoom_range = 0.2, 
        width_shift_range=0.1, 
        height_shift_range=0.1, 
        horizontal_flip = True, 
        vertical_flip=False)  

# Fits the data generator to some sample data.
datagen.fit(x_train)
#  Use Convolutional Neural Networks (CNN) Sequential model
# the first layer can receive an `input_shape` argument.
model = Sequential()
model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(224,224,3)))
model.add(MaxPool2D())

model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(2, activation="softmax"))

model.summary()
##  use compile() method: specifying a loss, metrics, and an optimizer. 

opt = Adam(lr=0.000001)
# builds the model and set the number of epochs equls 30 .
model.compile(optimizer = opt , loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])

history = model.fit(x_train,y_train,epochs = 30 , validation_data = (x_val, y_val))
# Get result (precision , recall , f1-score)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(30)

# plot the Training and Validation Accuracy
plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')


# Training and Validation Loss
plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
# Get result (precision , recall , f1-score)
predict_x=model.predict(x_val) 
classes_x=np.argmax(predict_x,axis=1)
print(classification_report(y_val, classes_x, target_names = ['withoutmask (Class 0)','mask (Class 1)']))