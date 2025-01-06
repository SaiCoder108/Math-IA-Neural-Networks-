import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print("Welcome to the NeuralNine (c) Handwritten Digits Recognition v0.1")

# Decide if to load an existing model or to train a new one
train_new_model = False 
ask = "give-weights"

if train_new_model:
    # Loading the MNIST data set with samples and splitting it
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalizing the data (making length = 1)
    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    # Create a neural network model
    model = tf.keras.models.Sequential()
    # Add one flattened input layer for the pixels
    model.add(tf.keras.layers.Flatten())
    # Add two dense hidden layers
    model.add(tf.keras.layers.Dense(units=16, activation='relu'))
    model.add(tf.keras.layers.Dense(units=16, activation='relu'))
    # Add one dense output layer for the 10 digits
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

    # Compiling and optimizing model
    model.compile(optimizer='adam', loss='MeanSquaredError', metrics=['accuracy'])

    # Training the model
    model.fit(X_train, y_train, epochs=6)

    # Evaluating the model
    val_loss, val_acc = model.evaluate(X_test, y_test)
    print(val_loss)
    print(val_acc)

    # Saving the model
    model.save('handwritten_digits.model1.keras')
elif ask == "give-weighs":
    model = tf.keras.models.load_model('handwritten_digits.model.keras')

    # Get the weights and biases
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Dense):  # Only export Dense layers
            weights, biases = layer.get_weights()

            # Save weights and biases to CSV files
            np.savetxt(f"layer_{i}_weights.csv", weights, delimiter=",", header="Weights")
            np.savetxt(f"layer_{i}_biases.csv", biases, delimiter=",", header="Biases")
else:
    # Load the model
    model = tf.keras.models.load_model('handwritten_digits.model.keras')

    print("hello?")
    # Load custom images and predict them
    image_number = 1
    while os.path.isfile('digits/digit{}.png'.format(image_number)): # not being fulfilled check why
        try:
            img = cv2.imread('digits/digit{}.png'.format(image_number))[:,:,0]
            img = np.invert(np.array([img]))  # Make sure image is in the correct format
            prediction = model.predict(img)
            print("The number is probably a {}".format(np.argmax(prediction)))
            plt.imshow(img[0], cmap=plt.cm.binary)
            plt.show()
            image_number += 1
        except:
            print("Error reading image! Proceeding with next image...")
            image_number += 1
