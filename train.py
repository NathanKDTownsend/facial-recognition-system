import numpy as np
from tensorflow.keras import models, layers, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from load_data import load_dataset
import random
import tensorflow as tf

# Load all images and labels from the Training folder
dataset_path = "/Users/yesiagreethx/Desktop/Facial_Recognition_Dataset"
X, y_cat, class_names = load_dataset(dataset_path)

# Split into 80% training and 20% temporary set, then split temp 50/50 into val and test
# Stratify ensures each split maintains the same class distribution
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_cat, test_size=0.2, stratify=np.argmax(y_cat, axis=1), random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=np.argmax(y_temp, axis=1), random_state=42
)

# Apply augmentation to artificially expand the training set and reduce overfitting
# Brightness excluded as it can push normalised pixel values outside [0, 1]
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
datagen.fit(X_train)

# Custom perceptron layer implementing the modified net input formula:
# net = w0*x0 + w0*x0^2 + w1*x1 + w1*x1^2 + ... + wn*xn + wn*xn^2
# Followed by sigmoid activation: sigma(net) = 1 / (1 + e^-net)
class CustomPerceptronLayer(layers.Layer):
    def __init__(self, units, **kwargs):
        super(CustomPerceptronLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        # Initialise shared weight matrix and bias using Glorot uniform
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
        # Linear term: w*x
        linear = tf.matmul(inputs, self.w)
        # Quadratic term: w*x^2 — same weights applied to squared inputs
        quadratic = tf.matmul(tf.square(inputs), self.w)
        net = linear + quadratic + self.b
        # Apply modified sigmoid activation
        return tf.sigmoid(net)

    def get_config(self):
        # Required for saving and loading the model with this custom layer
        config = super().get_config()
        config.update({'units': self.units})
        return config

# Build the CNN model with configurable hyperparameters
# Architecture: Conv blocks -> Flatten -> CustomPerceptronLayer -> Softmax output
def build_model(conv_layers=2, conv_filters=32, dense_units=256, learning_rate=0.001):
    model = models.Sequential([
        layers.Input(shape=(48, 48, 1)),
    ])

    for i in range(conv_layers):
        # Double filters with each block to capture increasingly complex features
        filters = conv_filters * (2 ** i)
        model.add(layers.Conv2D(filters, (3, 3), activation='relu', padding='same'))
        model.add(layers.Conv2D(filters, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
        # Dropout after pooling to prevent overfitting
        model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())

    # Replace standard Dense layer with custom perceptron using modified sigmoid
    model.add(CustomPerceptronLayer(dense_units))
    model.add(layers.Dropout(0.4))

    # Softmax output produces probability distribution across all six emotion classes
    model.add(layers.Dense(len(class_names), activation='softmax'))

    # SGD with momentum as specified — momentum helps smooth convergence
    optimizer = optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Genetic Algorithm search for best hyperparameters
# Each candidate is trained briefly on a small subset to estimate fitness
population_size = 3
generations = 2
best_score = 0
best_params = None
best_model = None

# Use a subset for speed during GA evaluation
subset_size = min(500, len(X_train))
X_subset = X_train[:subset_size]
y_subset = y_train[:subset_size]

for gen in range(1, generations + 1):
    print(f"\n--- Generation {gen} ---")
    for i in range(1, population_size + 1):
        # Randomly sample hyperparameter combination for this candidate
        conv_layers = random.choice([1, 2])
        conv_filters = random.choice([32, 64])
        dense_units = random.choice([128, 256])
        learning_rate = random.choice([0.001, 0.0005])

        print(f"Training model {i} with conv_layers={conv_layers}, conv_filters={conv_filters}, "
              f"dense_units={dense_units}, lr={learning_rate}")

        model = build_model(conv_layers, conv_filters, dense_units, learning_rate)

        # Train for 3 epochs on subset — enough to estimate relative performance
        history = model.fit(
            datagen.flow(X_subset, y_subset, batch_size=32),
            validation_data=(X_val[:subset_size], y_val[:subset_size]),
            epochs=3,
            verbose=1
        )

        # Use best validation accuracy across epochs as fitness score
        val_acc = max(history.history['val_accuracy'])
        print(f"Model {i} val_accuracy: {val_acc:.4f}")

        # Keep track of the best performing configuration
        if val_acc > best_score:
            best_score = val_acc
            best_params = (conv_layers, conv_filters, dense_units, learning_rate)
            best_model = model

print(f"\nBest GA model val_accuracy: {best_score:.4f}")
print(f"Best hyperparameters: conv_layers={best_params[0]}, conv_filters={best_params[1]}, "
      f"dense_units={best_params[2]}, learning_rate={best_params[3]}")

# Train final model on full training set using best hyperparameters from GA
final_model = build_model(*best_params)

# Halve learning rate if validation loss stops improving for 5 epochs
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
)

# Stop training early and restore best weights if val accuracy plateaus for 10 epochs
early_stop = callbacks.EarlyStopping(
    monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1
)

history = final_model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=40,
    callbacks=[reduce_lr, early_stop],
    verbose=1
)

# Evaluate on validation set — used to monitor generalisation during training
y_val_pred = final_model.predict(X_val)
y_val_classes = np.argmax(y_val_pred, axis=1)
y_val_true = np.argmax(y_val, axis=1)

print("\n--- Validation Performance ---")
print(f"Accuracy:  {accuracy_score(y_val_true, y_val_classes):.4f}")
print(f"Precision: {precision_score(y_val_true, y_val_classes, average='macro', zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_val_true, y_val_classes, average='macro', zero_division=0):.4f}")
print(f"F1 Score:  {f1_score(y_val_true, y_val_classes, average='macro', zero_division=0):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_val_true, y_val_classes))

# Evaluate on test set — final unseen evaluation of trained model
y_test_pred = final_model.predict(X_test)
y_test_classes = np.argmax(y_test_pred, axis=1)
y_test_true = np.argmax(y_test, axis=1)

print("\n--- Test Performance ---")
print(f"Accuracy:  {accuracy_score(y_test_true, y_test_classes):.4f}")
print(f"Precision: {precision_score(y_test_true, y_test_classes, average='macro', zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test_true, y_test_classes, average='macro', zero_division=0):.4f}")
print(f"F1 Score:  {f1_score(y_test_true, y_test_classes, average='macro', zero_division=0):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test_true, y_test_classes))

# Save the trained model in Keras format for use in test.py
final_model.save("facial_model.keras")
print("Final model saved as facial_model.keras")