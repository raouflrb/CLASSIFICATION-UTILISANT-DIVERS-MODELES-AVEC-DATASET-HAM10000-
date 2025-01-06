import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time

# Read metadata
skin_df = pd.read_csv('C:/Users/meker/Downloads/hm/HAM10000_metadata.csv')

# Define image size
SIZE = 32

# Label encoding to numeric values from text
le = LabelEncoder()
le.fit(skin_df['dx'])
skin_df['label'] = le.transform(skin_df["dx"]) 

# Read images based on image ID from the CSV file
image_path = {os.path.splitext(os.path.basename(x))[0]: x
              for x in glob(os.path.join('C:/Users/meker/Downloads/hm/', '*', '*.jpg'))}

# Define a function to load images with error handling
def load_image(file_path):
    try:
        img = Image.open(file_path)
        img = img.resize((SIZE, SIZE))
        return np.asarray(img)
    except Exception as e:
        return None
    
start_time = time.time()
# Define image paths and load images
skin_df['path'] = skin_df['image_id'].map(image_path.get)
skin_df['image'] = skin_df['path'].map(load_image)

# Prepare data for modeling
X = np.asarray(skin_df['image'].tolist()) / 255.0
Y = skin_df['label']

# Split data into training and testing sets using stratified sampling
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42, stratify=Y)

# Instantiate the Naive Bayes classifier
naive_bayes_model = GaussianNB()

# Measure execution time


# Train the model
naive_bayes_model.fit(x_train.reshape(x_train.shape[0], -1), y_train)

# Evaluate the model
accuracy = naive_bayes_model.score(x_test.reshape(x_test.shape[0], -1), y_test)
print("Test accuracy:", accuracy)

# Make predictions
y_pred = naive_bayes_model.predict(x_test.reshape(x_test.shape[0], -1))

# Print confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)

# Print classification report
report = classification_report(y_test, y_pred, target_names=le.classes_)
print("Classification report:\n", report)

# Calculate and print execution time
execution_time = time.time() - start_time
print("Execution time: {:.2f} seconds".format(execution_time))
