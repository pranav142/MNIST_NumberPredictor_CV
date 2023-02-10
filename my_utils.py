import matplotlib.pyplot as plt
import numpy as np


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