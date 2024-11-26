# Import necessary functions and classes from the image_classification module
from image_classification import *

# Load the previously trained model from the 'image_classifier.h5' file
model = models.load_model('image_classifier.h5')

# Prompt the user to input the path of the image they want to classify
path = input('Provide path of the image:')

# Read the image from the provided path using OpenCV (cv2)
img = cv.imread(path)

# Convert the image from BGR (OpenCV default) to RGB color format
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Preprocess the image by normalizing its pixel values to the range [0, 1]
# The model expects input in this range for prediction
prediction = model.predict(np.array([img])/255)

# Get the index of the highest predicted probability (the most likely class)
index = np.argmax(prediction)

# Define a list of class names corresponding to the CIFAR-10 dataset labels
class_names = ['Plane', 'Car', 'Bird', 'Deer', 'Dog', 'Horse', 'Ship', 'Truck']

# Print the predicted class name based on the highest probability index
print(f'Prediction is {class_names[index]}')