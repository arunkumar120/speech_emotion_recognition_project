# Speech Emotion Recognition (SER) Project
# Import necessary libraries
import kagglehub
import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
import pickle

# Step 1: Download the dataset using kagglehub
print("Downloading dataset...")
path = kagglehub.dataset_download("ejlok1/toronto-emotional-speech-set-tess")
print("Path to dataset files:", path)

# Step 2: Initialize variables for storing features and labels
audio_data = []
labels = []

# Step 3: Traverse the dataset and extract features
print("Processing dataset and extracting features...")
for root, dirs, files in os.walk(path):
    for file in tqdm(files, desc="Processing files"):
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            try:
                # Load audio file
                y, sr = librosa.load(file_path, sr=None)
                
                # Extract MFCC features
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfccs_mean = np.mean(mfccs.T, axis=0)  # Calculate mean for fixed-length features
                
                # Append features and corresponding label
                audio_data.append(mfccs_mean)
                labels.append(os.path.basename(root))  # Use folder name as label
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

# Step 4: Verify data collection
print(f"Number of audio samples processed: {len(audio_data)}")
print(f"Number of labels collected: {len(labels)}")

if len(audio_data) == 0:
    raise ValueError("No audio data was processed. Check the dataset structure and file formats.")

# Step 5: Convert data to NumPy arrays
X = np.array(audio_data)
y = np.array(labels)

# Step 6: Encode labels using LabelEncoder
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
print("Classes detected:", encoder.classes_)

# Step 7: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# Step 8: Train a Support Vector Machine (SVM) classifier
print("Training the SVM classifier...")
svm = SVC(kernel='linear', C=1.0, random_state=42)
svm.fit(X_train, y_train)

# Step 9: Evaluate the model
print("Evaluating the model...")
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# Step 10: Save the trained model for future use
model_path = "emotion_svm_model.pkl"
with open(model_path, "wb") as model_file:
    pickle.dump(svm, model_file)
print(f"Model saved as {model_path}")

# Step 11: Example Prediction (Optional)
print("Testing with a sample prediction...")
sample_index = 0
sample_features = X_test[sample_index].reshape(1, -1)
predicted_label = encoder.inverse_transform(svm.predict(sample_features))
true_label = encoder.inverse_transform([y_test[sample_index]])
print(f"Predicted Emotion: {predicted_label[0]}, True Emotion: {true_label[0]}")
