import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
 
mnist = tf.keras.datasets.mnist
 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
# 
# model = tf.keras.models.Sequential()
# # flattens the input from a 28x28 grid into a 784 line of pixels.
# model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
# # after a flatten layer, we want a dense layer
# model.add(tf.keras.layers.Dense(128, activation="relu"))
# model.add(tf.keras.layers.Dense(128, activation="relu"))
# # output layer, ints 0-9, softmax is a pdf, returns an array of probs
# model.add(tf.keras.layers.Dense(10, activation="softmax"))
# 
# model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'])
# 
# model.fit(x_train, y_train, epochs=10)
# 
# model.save("handwritten.keras")
# 
model = tf.keras.models.load_model("./handwritten.keras")

loss, accuracy = model.evaluate(x_test, y_test)

print(loss)
print(accuracy)

image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        image_path = f"digits/digit{image_number}.png"
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"Could not read {image_path}")

        # The exported MNIST images already have white digits on a black
        # background, so they should not be inverted.
        img = img.astype("float32") / 255.0
        img = np.array([img])

        prediction = model.predict(img, verbose=0)
        predicted_digit = np.argmax(prediction[0])
        print(f"{image_path}: this digit is probably a {predicted_digit}")

    except Exception as error:
        print(f"Could not classify digit{image_number}.png: {error}")

    image_number += 1
