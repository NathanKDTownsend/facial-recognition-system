def load_dataset(base_path, img_size=(48,48)):
    """
    Load images and labels from dataset folder structure:
    base_path/
        Training/
            Angry/
            Fear/
            ...
        Validation/
            Angry/
            ...
        Testing/
            ...
    """
    import os, cv2, numpy as np
    from tensorflow.keras.utils import to_categorical

    # Define expected dataset subsets (only Training is loaded for model training)
    subsets = ["Training", "Validation", "Testing"]
    X, y = [], []

    # Get sorted class names from Training folder, skipping system files like .DS_Store
    class_names = sorted([d for d in os.listdir(os.path.join(base_path, "Training")) 
                          if os.path.isdir(os.path.join(base_path, "Training", d))])

    for idx, class_name in enumerate(class_names):
        print(f"Loading class {class_name} with label {idx}...")
        class_dir = os.path.join(base_path, "Training", class_name)

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)

            # Load image as grayscale — colour is not needed for expression recognition
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Skip corrupted or unreadable files
            if img is None:
                continue

            # Resize to consistent dimensions required by the network input layer
            img = cv2.resize(img, img_size)

            # Normalise pixel values from [0, 255] to [0, 1] for stable training
            img = img.astype('float32') / 255.0

            X.append(img)
            y.append(idx)

    # Convert lists to numpy arrays for use with Keras
    X = np.array(X)
    y = np.array(y)

    # Add channel dimension: (samples, height, width) -> (samples, height, width, 1)
    X = X.reshape(-1, img_size[0], img_size[1], 1)

    # Convert integer labels to one-hot encoded vectors for categorical cross-entropy
    y_cat = to_categorical(y, num_classes=len(class_names))

    print(f"Dataset loaded! Total images: {len(X)}")
    print(f"Number of classes: {len(class_names)}")

    return X, y_cat, class_names