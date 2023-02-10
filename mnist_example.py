import tensorflow
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, Flatten, GlobalAvgPool2D

# Three Approaches to building deep learning models
# 1. Tensorflow.keras.sequential
# 2. Functional Approach => Function that returns a model
# 3. Tensorflow.keras.models: Inherit a model from a class

model = tensorflow.keras.Sequential(
    [
        Input(shape=(28,28,1)),
        Conv2D(32, (3,3), activation = 'relu'),
        Conv2D(64, (3,3), activation = 'relu'),
        MaxPool2D(),
        BatchNormalization(),

        Conv2D(128, (3,3), activation = 'relu'),
        MaxPool2D(),
        BatchNormalization(),

        GlobalAvgPool2D(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ]
)

def display_some_examples(examples, labels):

    # sets figure window size
    plt.figure(figsize=(10,10),)

    for i in range(25):

        # random index value between 0 and max number of samples
        idx = np.random.randint(0, examples.shape[0]-1)
        
        img = examples[idx]
        label = labels[idx]
        
        # creates subplots in a 5,5 grid
        plt.subplot(5,5, i+1)
        plt.title(str(label))
        # right layout gives even and clear spacing
        plt.tight_layout()
        # makes images gray scale
        plt.imshow(img, cmap='gray')
    
    plt.show()

def predict_some_examples(examples, labels, predictions):

    plt.figure(figsize=(10,10))

    for i in range(25):

        idx = np.random.randint(0, examples.shape[0]-1)

        img = examples[idx]
        label = labels[idx]

        prediction = np.argmax(predictions[idx])
        plt.subplot(5,5, i+1)
        title = str(prediction) + " Actual: " + str(np.argmax(label))
        plt.title(title)
        plt.tight_layout()
        plt.imshow(img, cmap='gray')
    
    plt.show()

if __name__ == '__main__':

    # loads the train and test data
    # the x is the images and the y is the labels for the images, (60000, 28, 28) => 600000 images, 28 x 28 pixels
    (x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()

    print("X_train.shape= ", x_train.shape)
    print("y_train.shape= ", y_train.shape)
    print("X_test.shape= ", x_test.shape)
    print("y_test.shape= ", y_test.shape)
    
    #display_some_examples(x_train, y_train)

    # normalizes the data, takes pixel values and puts them on scale 0 to 1
    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255

    # one hot encodes the train and test data labels
    y_train = tensorflow.keras.utils.to_categorical(y_train, 10)
    y_test = tensorflow.keras.utils.to_categorical(y_test, 10)

    # adds a dimension because model takes (28, 28, 1) but data is originally (28,28)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # use sparse_categorical_cross_entropy when the labels are not one hot encoded
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')

    # validation set will see how well the model is doing with each epoch
    model.fit(x_train, y_train, batch_size=64, epochs=3, validation_split=0.2)

    model.evaluate(x_test,y_test, batch_size=64)

    predictions = model.predict(x_test)

    predict_some_examples(x_test, y_test, predictions)
    