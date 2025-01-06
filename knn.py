import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import time


np.random.seed(42)

# Read metadata
skin_df = pd.read_csv('C:/Users/meker/Downloads/hm/HAM10000_metadata.csv')

# Define image size
SIZE = 32

# Label encoding to numeric values from text
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(skin_df['dx'])
skin_df['label'] = le.transform(skin_df["dx"]) 

# Data distribution visualization
fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(221)
skin_df['dx'].value_counts().plot(kind='bar', ax=ax1)
ax1.set_ylabel('Count')
ax1.set_title('Cell Type')

ax2 = fig.add_subplot(222)
skin_df['sex'].value_counts().plot(kind='bar', ax=ax2)
ax2.set_ylabel('Count', size=15)
ax2.set_title('Sex')

ax3 = fig.add_subplot(223)
skin_df['localization'].value_counts().plot(kind='bar')
ax3.set_ylabel('Count',size=12)
ax3.set_title('Localization')

ax4 = fig.add_subplot(224)
sample_age = skin_df[pd.notnull(skin_df['age'])]
sns.distplot(sample_age['age'], fit=stats.norm, color='red')
ax4.set_title('Age')

plt.tight_layout()
plt.show()

# Distribution of data into various classes 
print(skin_df['label'].value_counts())



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
        return np.asarray(img)
    except Exception as e:
        return None

start_time = time.time()


# Define image paths and load images
df_balanced['path'] = df_balanced['image_id'].map(image_path.get)
df_balanced['image'] = df_balanced['path'].map(load_image)

# Remove rows with None values (corresponding to failed image loading)
df_balanced = df_balanced.dropna()

# Prepare data for modeling
X = np.asarray(df_balanced['image'].tolist()) / 255.0
Y = df_balanced['label']
Y_cat = to_categorical(Y, num_classes=7)

# Filter out classes with very few samples
class_counts = df_balanced['label'].value_counts()
valid_classes = class_counts[class_counts >= 100].index
df_balanced_filtered = df_balanced[df_balanced['label'].isin(valid_classes)]

# Prepare data for modeling
X_filtered = np.asarray(df_balanced_filtered['image'].tolist()) / 255.0
Y_filtered = df_balanced_filtered['label']
Y_cat_filtered = to_categorical(Y_filtered, num_classes=7)

# Split data into training and testing sets using stratified sampling
x_train, x_test, y_train, y_test = train_test_split(X_filtered, Y_cat_filtered, test_size=0.1, random_state=42, stratify=Y_cat_filtered)

# Reshape data for KNN
x_train_flattened = x_train.reshape(x_train.shape[0], -1)
x_test_flattened = x_test.reshape(x_test.shape[0], -1)

# Initialize and train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)  # You can adjust the number of neighbors as needed
knn.fit(x_train_flattened, y_train)

# Evaluate the KNN model
accuracy = knn.score(x_test_flattened, y_test)

# Start time

# Print accuracy
print('Test accuracy:', accuracy)

# Predictions on test data
y_pred_knn = knn.predict(x_test_flattened)

# Print classification report
print("Classification Report:")
print(classification_report(y_test.argmax(axis=1), y_pred_knn.argmax(axis=1)))

# Calculate and print precision, recall, and F1 score
precision = precision_score(y_test.argmax(axis=1), y_pred_knn.argmax(axis=1), average='weighted')
recall = recall_score(y_test.argmax(axis=1), y_pred_knn.argmax(axis=1), average='weighted')
f1 = f1_score(y_test.argmax(axis=1), y_pred_knn.argmax(axis=1), average='weighted')
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Calculate and print error
error = 1 - accuracy
print("Error:", error)

# End time
end_time = time.time()

# Print execution time
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")

# Print confusion matrix for KNN
cm_knn = confusion_matrix(y_test.argmax(axis=1), y_pred_knn.argmax(axis=1))
fig, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(cm_knn, annot=True, linewidths=.5, ax=ax)
plt.show()
