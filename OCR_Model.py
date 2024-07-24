import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import cv2
import numpy as np

#Data partition for training and testing
train_datagen=ImageDataGenerator(rescale=1./255, validation_split=0.3)
test_datagen=ImageDataGenerator(rescale=1./255)
train_generator=train_datagen.flow_from_directory("captcha_data_2k",batch_size=48,class_mode="categorical", seed=42)
test_generator=test_datagen.flow_from_directory("captcha_data_2k",batch_size=48,class_mode="categorical", seed=42)

#CNN Model
model=Sequential()
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(3, activation="softmax"))

#Compiling the model
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

#Training the model
model.fit(train_generator,epochs=2,validation_data=test_generator)

#Testing with own Input
new_image = cv2.imread("295431.jpg")
cv2.imshow("Input image",new_image)
#print(new_image.shape)
resized_image = cv2.resize(new_image,(28, 28))
#print(resized_image)
new_image = resized_image.reshape((1, 28, 28, 1))
#print(new_image)
predictions = model.predict(new_image)
predicted_digit = np.argmax(predictions[0])
print("Captcha Text:", predicte_digit)

#Validation
test_loss, test_acc = model.evaluate(test_generator)
print("Test Accuracy:", round(test_acc*100))
