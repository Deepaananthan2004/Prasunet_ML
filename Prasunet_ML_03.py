import os
import numpy as np
from skimage.feature import hog
from skimage import io, transform, color
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import imageio
# Define the categories
data_dir = 'D:\\training_set'
categories = ['cats', 'dogs']
hog_features = []
target = []
# Load and preprocess the data
for category in categories:
    print(f'Loading {category}...')
    path = os.path.join(data_dir, category)
    if not os.path.exists(path):
        print(f'Path {path} does not exist.')
        continue
    
    for img_file in os.listdir(path):
        try:
            img_path = os.path.join(path, img_file)
            img_array = io.imread(img_path)
            img_resized = transform.resize(img_array, (150, 150, 3))
            gray_img = color.rgb2gray(img_resized)
            hog_feat = hog(gray_img, orientations=9, pixels_per_cell=(10, 10), cells_per_block=(2, 2))
            hog_features.append(hog_feat)
            target.append(categories.index(category))
        except Exception as e:
            print(f'Error loading image {img_file}: {str(e)}')
    
    print(f'Successfully loaded {category}!')
# Convert lists to numpy arrays
hog_features = np.array(hog_features)
target = np.array(target)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(hog_features, target, test_size=0.2, random_state=42)
# Train the SVM model
model = svm.SVC(kernel='linear', C=1)
model.fit(X_train, y_train)
# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))