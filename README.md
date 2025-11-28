# Deep-Learning-Exp3

**DL-Convolutional Deep Neural Network for Image Classification**

**AIM**

To develop a convolutional neural network (CNN) classification model for the given dataset.

**THEORY**

The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28×28 pixels. The task is to classify these images into their respective digit categories. CNNs are particularly well-suited for image classification tasks as they can automatically learn spatial hierarchies of features through convolutional layers, pooling layers, and fully connected layers.

**Neural Network Model**

<img width="785" height="440" alt="image" src="https://github.com/user-attachments/assets/55b56bdd-ee06-44f2-b52a-3a6c4aa60efd" />


**DESIGN STEPS**

STEP 1:

Preprocess the CIFAR-10 dataset by scaling pixel values to [0, 1] and applying one-hot encoding to labels.

STEP 2:

Build a CNN model with the following layers:

Input → Convolution → MaxPooling → Convolution → MaxPooling → Flatten → Dense layers → Output.

STEP 3:

Compile the model using categorical cross-entropy loss and the Adam optimizer.

STEP 4:

Train the CNN for multiple epochs (e.g., 10) with a batch size of 64.

STEP 5:

Evaluate the model on the test set by analyzing accuracy, loss curves, confusion matrix, and classification report. Test predictions on custom images.

**PROGRAM**

Name: HARISH GOWTHAM E

Register Number:2305002009
~~~
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape

single_image= X_train[0]
single_image.shape

plt.imshow(single_image,cmap='gray')
y_train.shape

X_train.min()
X_train.max()

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

X_train_scaled.min()
X_train_scaled.max()

y_train[0]

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

type(y_train_onehot)

y_train_onehot.shape

single_image = X_train[500]
plt.imshow(single_image,cmap='gray')

y_train_onehot[500]

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train_scaled ,y_train_onehot, epochs=10,
          batch_size=128,
          validation_data=(X_test_scaled,y_test_onehot))

metrics = pd.DataFrame(model.history.history)
metrics.head()

metrics[['accuracy','val_accuracy']].plot()

metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print(confusion_matrix(y_test,x_test_predictions))

print(classification_report(y_test,x_test_predictions))

img = image.load_img('7.png')

type(img)

img = image.load_img('7.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)

plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0

x_single_prediction = np.argmax(model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),axis=1)

print(x_single_prediction)
~~~

**OUTPUT**

**Training Loss per Epoch**

<img width="796" height="201" alt="image" src="https://github.com/user-attachments/assets/7c9cc674-e0b1-4577-afab-39c3af4d35b1" />


<img width="684" height="500" alt="image" src="https://github.com/user-attachments/assets/49872e8e-000a-4ce3-986c-66bf92784ca1" />

<img width="684" height="495" alt="image" src="https://github.com/user-attachments/assets/bf9db1bb-7125-4f1e-97e5-c0aac7bd6e47" />


**Confusion Matrix**

<img width="668" height="261" alt="image" src="https://github.com/user-attachments/assets/486ebb28-c536-443e-98e2-9ebe163e77b7" />


**Classification Report**

<img width="496" height="497" alt="image" src="https://github.com/user-attachments/assets/76f80122-3a71-4a25-afa6-ca083b6a7460" />


**New Sample Data Prediction**

<img width="680" height="411" alt="image" src="https://github.com/user-attachments/assets/6a30e085-1bca-4307-83e7-89782692b505" />

<img width="509" height="499" alt="image" src="https://github.com/user-attachments/assets/af55e404-be8f-43b6-852a-4fabd887756c" />



**RESULT**

Thus, a Convolutional Deep Neural Network for digit classification was developed and successfully verified with both MNIST dataset and scanned handwritten images.
