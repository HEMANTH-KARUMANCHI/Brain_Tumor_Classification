# brain_tumor_mobilenetv2.py

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set the path to your dataset
train_path = r'C:\Users\Hemanth\Documents\Projects_2025\Brain_Tumor_D\dataset\train'
val_path = r'C:\Users\Hemanth\Documents\Projects_2025\Brain_Tumor_D\dataset\val'

# Ensure output directories exist
output_dir = r'C:\Users\Hemanth\Documents\Projects_2025\Brain_Tumor_D\outputs'
vis_dir = r'C:\Users\Hemanth\Documents\Projects_2025\Brain_Tumor_D\Visualizations'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(vis_dir, exist_ok=True)

# Parameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10

# 1. Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.2,
    shear_range=0.2,
    rotation_range=20,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# 2. Load MobileNetV2 and customize
base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# 3. Train the Model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# 4. Save outputs
loss, acc = model.evaluate(val_generator)
with open(os.path.join(output_dir, 'model_accuracy.txt'), 'w') as f:
    f.write(f"Validation Accuracy: {acc * 100:.2f}%\n")

with open(os.path.join(output_dir, 'model_loss.txt'), 'w') as f:
    f.write(f"Validation Loss: {loss:.4f}\n")

# 5. Visualize Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.savefig(os.path.join(vis_dir, 'accuracy_plot.png'))
plt.close()

# 6. Visualize Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.savefig(os.path.join(vis_dir, 'loss_plot.png'))
plt.close()

# 7. Confusion Matrix
Y_pred = model.predict(val_generator)
y_pred = np.argmax(Y_pred, axis=1)
cm = confusion_matrix(val_generator.classes, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=val_generator.class_indices.keys(),
            yticklabels=val_generator.class_indices.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(vis_dir, 'confusion_matrix.png'))
plt.close()
