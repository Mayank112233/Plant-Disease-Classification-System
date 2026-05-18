"""
Plant Disease Classification System
-----------------------------------
Task: Classify plant leaf diseases (38 classes) using PlantVillage dataset
Model: EfficientNetB0 with Transfer Learning

Instructions for Beginners:
1. Prerequisites:
   You need to have Python installed. It's recommended to use a virtual environment.
   
2. Installation of Required Libraries:
   Open your terminal/command prompt and run the following command:
   pip install tensorflow tensorflow-datasets numpy matplotlib scikit-learn seaborn

3. How to Run:
   Run this script directly from your terminal:
   python plant_disease_classifier.py

   The script will:
   - Automatically download the PlantVillage dataset (~800 MB).
   - Prepare the data and build the model.
   - Train the model (this can take some time depending on your CPU/GPU).
   - Save the trained model as 'plant_disease_model.keras'.
   - Generate evaluation plots and confusion matrix.
"""

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

# ==========================================
# CONFIGURATION
# ==========================================
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 50
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = 'plant_disease_model.keras'

print(f"TensorFlow Version: {tf.__version__}")

# ==========================================
# 1. DATA PREPARATION
# ==========================================
print("\n--- 1. DATA PREPARATION ---")
print("Loading PlantVillage dataset from TensorFlow Datasets...")

# Load the dataset
print("Loading dataset from local folder...")

dataset = tf.keras.utils.image_dataset_from_directory(
    "PlantVillage",
    validation_split=0.3,
    subset="both",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)
train_ds, val_ds = dataset

CLASS_NAMES = train_ds.class_names
NUM_CLASSES = len(CLASS_NAMES)

print(f"Total Classes: {NUM_CLASSES}")

# Create test dataset
val_batches = tf.data.experimental.cardinality(val_ds)

test_ds = val_ds.take(val_batches // 2)
val_ds = val_ds.skip(val_batches // 2)

# Data Preprocessing & Augmentation Pipeline
# We use Sequential API for augmentation to apply it inside the dataset pipeline or model
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomBrightness(factor=0.2),
], name="data_augmentation")


    
def preprocess_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Prepare Training Dataset
print("Configuring data pipelines...")

train_ds = train_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)

# Prepare Validation Dataset
val_ds = val_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

# Prepare Test Dataset
test_ds = test_ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)

# ==========================================
# 2. MODEL BUILDING
# ==========================================
print("\n--- 2. MODEL BUILDING ---")
print("Loading pre-trained EfficientNetB0...")

# Load pre-trained EfficientNetB0 without the top classification layers
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False, 
    weights='imagenet',
    input_shape=IMG_SIZE + (3,) # (224, 224, 3)
)

# Freeze the base model layers so they don't get updated during early training
base_model.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))

x = data_augmentation(inputs)

x = base_model(x, training=False)

# Add custom dense layers on top
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(256, activation='relu', name='custom_dense_256')(x)
x = tf.keras.layers.Dropout(0.5, name='dropout_regularization')(x)

# Output layer with 38 units (for 38 classes) and Softmax activation
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

model = tf.keras.Model(inputs, outputs, name='Plant_Disease_EfficientNetB0')

print(model.summary())

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================================
# 3. TRAINING
# ==========================================
print("\n--- 3. TRAINING ---")

# Define callbacks for training
callbacks = [
    # Stop training early if validation loss stops improving for 5 epochs
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    # Reduce learning rate if validation loss plateaus
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

print(f"Starting training for up to {EPOCHS} epochs...")
print("NOTE: Training might take a long time on a CPU.")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Save the trained model
model.save(MODEL_SAVE_PATH)
print(f"\nModel saved successfully to: {MODEL_SAVE_PATH}")


# ==========================================
# 4. EVALUATION
# ==========================================
print("\n--- 4. EVALUATION ---")

# Evaluate on test set
print("Evaluating on Test Set...")
test_loss, test_acc, test_precision, test_recall = model.evaluate(test_ds)

# Calculate F1-Score manually from precision and recall
if (test_precision + test_recall) > 0:
    test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)
else:
    test_f1 = 0.0

print(f"\nTest Results:")
print(f"Accuracy:  {test_acc:.4f} (Target: >= 0.85)")
print(f"Loss:      {test_loss:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall:    {test_recall:.4f}")
print(f"F1-Score:  {test_f1:.4f}")

# Visualizing Training History (Loss & Accuracy)
plt.figure(figsize=(14, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig('training_history.png')
print("Saved training history plot as 'training_history.png'")
plt.show()

# Predictions for Confusion Matrix and Classification Report
print("\nGenerating Classification Report and Confusion Matrix...")
y_true = []
y_pred = []

for images, labels in test_ds:
    # Get true labels (argmax undoes the one-hot encoding)
    true_classes = labels.numpy()
    y_true.extend(true_classes)
    
    # Get predictions
    preds = model.predict(images, verbose=0)
    pred_classes = tf.argmax(preds, axis=1).numpy()
    y_pred.extend(pred_classes)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Classification Report
print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# Confusion Matrix
plt.figure(figsize=(20, 20))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=False, cmap='Blues', fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(np.arange(NUM_CLASSES) + 0.5, CLASS_NAMES, rotation=90)
plt.yticks(np.arange(NUM_CLASSES) + 0.5, CLASS_NAMES, rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Saved confusion matrix plot as 'confusion_matrix.png'")
plt.show()


# ==========================================
# 5. PREDICTIONS (Usage Example)
# ==========================================
print("\n--- 5. PREDICTIONS (Sample) ---")

def predict_new_image(image_tensor, true_label_name):
    """
    Function to predict disease from a preprocessed plant leaf image tensor.
    """
    # Add batch dimension since model expects (batch, height, width, channels)
    img_array = tf.expand_dims(image_tensor, 0)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_class_idx]
    confidence = 100 * np.max(predictions[0])
    
    print(f"Actual    : {true_label_name}")
    print(f"Predicted : {predicted_class}")
    print(f"Confidence: {confidence:.2f}%\n")

# Display a few sample predictions from the test set
print("Showing 3 sample predictions from the test set:\n")

# Get one batch from test_ds
for images, labels in test_ds.take(1):
    # Just take the first 3 images in the batch
    for i in range(3):
        img = images[i]
        true_idx = tf.argmax(labels[i]).numpy()
        true_class_name = CLASS_NAMES[true_idx]
        
        print(f"Sample #{i+1}")
        predict_new_image(img, true_class_name)

print("\nProject script completed successfully!")
