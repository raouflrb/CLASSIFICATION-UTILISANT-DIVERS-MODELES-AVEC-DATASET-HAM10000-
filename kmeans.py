import numpy as np
import pandas as pd
import os
from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import time

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
    
start_time = time.time()

# Define image paths and load images
df_balanced['path'] = df_balanced['image_id'].map(image_path.get)
df_balanced['image'] = df_balanced['path'].map(load_image)

# Remove rows with None values (corresponding to failed image loading)
df_balanced = df_balanced.dropna()

# Prepare data for modeling
X = np.asarray(df_balanced['image'].tolist())
Y = df_balanced['label']

# Split data into training and testing sets using stratified sampling
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42, stratify=Y)

# Use KMeans for feature extraction
n_clusters = 5  # You can experiment with different values
kmeans = KMeans(n_clusters=n_clusters, random_state=42)

x_train_features = kmeans.fit_transform(x_train)
x_test_features = kmeans.transform(x_test)

# Scale the features
scaler = StandardScaler()
x_train_features = scaler.fit_transform(x_train_features)
x_test_features = scaler.transform(x_test_features)

# Use Random Forest classifier instead of Logistic Regression
rf = RandomForestClassifier()  # Using default parameters
grid_search = GridSearchCV(rf, {}, cv=3, n_jobs=-1)  # No parameters to search
grid_search.fit(x_train_features, y_train)
best_rf = grid_search.best_estimator_

# Predict on the test data
y_pred = best_rf.predict(x_test_features)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print('Test accuracy:', accuracy)

# Print classification report
report = classification_report(y_test, y_pred)
print('Classification Report:\n', report)

# Print confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Calculate execution time
execution_time = time.time() - start_time
# Print execution time
print('Execution Time:', execution_time, 'seconds')
