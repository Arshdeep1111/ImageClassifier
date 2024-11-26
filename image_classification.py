# Import necessary libraries
import cv2 as cv  # OpenCV library for computer vision tasks (though it's not used in the current code)
import numpy as np  # NumPy library for numerical operations (not used directly in the code)
from tensorflow.keras import datasets, layers, models  # Keras modules from TensorFlow for data, layers, and model building

# Function to load and preprocess the CIFAR-10 dataset
def get_data():
    # Load CIFAR-10 dataset (a dataset of 60,000 32x32 color images in 10 classes)
    (training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
    
    # Normalize the image pixel values to a range of 0 to 1 by dividing by 255
    training_images, testing_images = training_images/255, testing_images/255
    
    # Optionally, limit the dataset size for faster training by taking a subset
    training_images = training_images[:20000]
    training_labels = training_labels[:20000]
    testing_images = testing_images[:4000]
    testing_labels = testing_labels[:4000]

# Function to build, train, and evaluate the CNN model
def train_model():
    # Initialize a Sequential model
    model = models.Sequential()
    
    # Add the first convolutional layer with 32 filters, a 3x3 kernel, and ReLU activation function
    # The input shape is (32, 32, 3) as CIFAR-10 images are 32x32 pixels with 3 color channels
    model.add(layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (32,32,3)))
    
    # Add the first max pooling layer to reduce the spatial dimensions of the feature map
    model.add(layers.MaxPooling2D(2,2))
    
    # Add the second convolutional layer with 64 filters and ReLU activation
    model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
    
    # Add another max pooling layer
    model.add(layers.MaxPooling2D(2,2))
    
    # Add a third convolutional layer with 64 filters and ReLU activation
    model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
    
    # Flatten the output from the previous layers to make it suitable for the fully connected layers
    model.add(layers.Flatten())
    
    # Add a fully connected layer with 64 units and ReLU activation
    model.add(layers.Dense(64, activation = 'relu'))
    
    # Add the output layer with 10 units (one for each CIFAR-10 class) and softmax activation for classification
    model.add(layers.Dense(10, activation = 'softmax'))
    
    # Compile the model with Adam optimizer, sparse categorical cross-entropy loss, and accuracy as a metric
    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    
    # Train the model on the training data for 10 epochs
    model.fit(training_images, training_labels, epochs = 10)
    
    # Evaluate the model's performance on the testing data
    loss, accuracy = model.evaluate(testing_images, testing_labels)
    
    # Print the loss and accuracy of the model on the test data
    print(f"Loss = {loss}")
    print(f'Accuracy = {accuracy}')
    
    # Save the trained model to a file for later use
    model.save('image_classifier.h5')

# Main execution block, called when the script is run
if __name__ == '__main__':
    # Call the function to load and preprocess the data
    get_data()
    
    # Call the function to build, train, and evaluate the model
    train_model()