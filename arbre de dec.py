import numpy as np
import pandas as pd
import os
from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from time import time

# Read metadata
skin_df = pd.read_csv('C:/Users/meker/Downloads/hm/HAM10000_metadata.csv')

# Define image size
SIZE = 32

# Label encoding to numeric values from text
le = LabelEncoder()
le.fit(skin_df['dx'])
skin_df['label'] = le.transform(skin_df["dx"]) 

# Balance data
df_balanced = skin_df.groupby('label').apply(lambda x: x.sample(n=2000, replace=True, random_state=42)).reset_index(drop=True)

# Check for classes with fewer samples than the desired test size
classes_with_few_samples = df_balanced['label'].value_counts()[df_balanced['label'].value_counts() < 100]

# Remove these classes from the data
df_balanced = df_balanced[~df_balanced['label'].isin(classes_with_few_samples.index)]

# Read images based on image ID from the CSV file
image_path = {os.path.splitext(os.path.basename(x))[0]: x
              for x in glob(os.path.join('C:/Users/meker/Downloads/hm/', '*', '*.jpg'))}

# Define a function to load images with error handling
def load_image(file_path):
    try:
        img = Image.open(file_path)
        img = img.resize((SIZE, SIZE))
        return np.asarray(img).flatten()  # Flatten the image
    except Exception as e:
        return None
    
    
start_time = time()

# Define image paths and load images
df_balanced['path'] = df_balanced['image_id'].map(image_path.get)
df_balanced['image'] = df_balanced['path'].map(load_image)

# Remove rows with None values (corresponding to failed image loading)
df_balanced = df_balanced.dropna()

# Prepare data for modeling
X = np.asarray(df_balanced['image'].tolist())
Y = df_balanced['label']

# Split data into training and testing sets using stratified sampling
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.17, random_state=42, stratify=Y)

# Scale the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Use Decision Tree classifier
dt = DecisionTreeClassifier(random_state=42)
parameters = {'max_depth': [None, 10, 20, 50],
              'min_samples_split': [2, 5, 10]}
grid_search = GridSearchCV(dt, parameters, cv=3, n_jobs=-1)
grid_search.fit(x_train, y_train)
best_dt = grid_search.best_estimator_

# Predict on the test data
y_pred = best_dt.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print('Test accuracy:', accuracy)

# Print confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification Report and Execution Time

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
execution_time = time() - start_time
print("Execution Time:", execution_time, "seconds")
