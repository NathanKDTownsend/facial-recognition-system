import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import random

# CustomPerceptronLayer must be defined here so Keras can reconstruct it when loading the model
# Without this, load_model raises a TypeError as it cannot locate the custom class
class CustomPerceptronLayer(layers.Layer):
    def __init__(self, units, **kwargs):
        super(CustomPerceptronLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        # Initialise shared weight matrix and bias
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='weights'
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )

    def call(self, inputs):
        # Compute linear and quadratic net input, then apply modified sigmoid
        linear = tf.matmul(inputs, self.w)
        quadratic = tf.matmul(tf.square(inputs), self.w)
        net = linear + quadratic + self.b
        return tf.sigmoid(net)

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config

# Load saved model — custom_objects required to deserialise the CustomPerceptronLayer
model = load_model(
    'facial_model.keras',
    custom_objects={'CustomPerceptronLayer': CustomPerceptronLayer},
    compile=False
)

# Six emotion classes matching the training label order
class_names = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Pre-process a single image to match the format used during training
def preprocess_image(img_path, img_size=(48,48)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size)
    # Histogram equalisation for contrast normalisation across varying lighting conditions
    img = cv2.equalizeHist(img)
    # Normalise to [0, 1] to match training data range
    img = img / 255.0
    # Add batch and channel dimensions: (48,48) -> (1,48,48,1)
    img = img[np.newaxis, ..., np.newaxis]
    return img

# Load all images from the Testing folder, organised by class subdirectory
testing_path = "Facial_Recognition_Dataset/Testing"
test_images = []
test_labels = []
for idx, expression in enumerate(class_names):
    folder_path = os.path.join(testing_path, expression)
    if not os.path.isdir(folder_path):
        continue
    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        img = preprocess_image(img_path)
        test_images.append(img)
        test_labels.append(idx)

# Stack individual image arrays into a single batch array
X_test = np.vstack(test_images)
y_test = np.array(test_labels)

# Run inference — returns probability vector per image
y_pred_probs = model.predict(X_test)
# Take the class with highest probability as the prediction
y_pred = np.argmax(y_pred_probs, axis=1)

# Compute evaluation metrics using weighted averaging across all six classes
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
cm = confusion_matrix(y_test, y_pred)

# Display a random sample of 6 test images with true and predicted labels
sample_indices = random.sample(range(len(X_test)), min(6, len(X_test)))
plt.figure(figsize=(12, 8))
for i, idx in enumerate(sample_indices):
    plt.subplot(2, 3, i+1)
    img = X_test[idx].reshape(48, 48)
    true_label = class_names[y_test[idx]]
    pred_label = class_names[y_pred[idx]]
    plt.imshow(img, cmap='gray')
    plt.title(f"True: {true_label}\nPred: {pred_label}")
    plt.axis('off')
plt.tight_layout()
plt.savefig('sample_predictions.png')

# Render confusion matrix heatmap to show per-class prediction distribution
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig('confusion_matrix.png')

# Print final evaluation metrics
print("\n--- Test Set Performance ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
