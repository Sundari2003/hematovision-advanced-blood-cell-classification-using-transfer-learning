import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ------------------ CONFIGURATION ------------------ #
data_dir = r"C:\Users\HematoVision\dataset"
save_dir = r"C:\Users\HematoVision\model"
img_size = (224, 224)
batch_size = 32
epochs = 10

# ------------------ PREPARE FOLDERS ------------------ #
os.makedirs(save_dir, exist_ok=True)

# ------------------ DATA GENERATORS ------------------ #
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# ------------------ SAVE CLASS LABELS ------------------ #
class_indices_path = os.path.join(save_dir, "class_indices.json")
with open(class_indices_path, "w") as f:
    json.dump(train_generator.class_indices, f)

# ------------------ MODEL DEFINITION ------------------ #
base_model = MobileNetV2(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# ------------------ TRAIN MODEL ------------------ #
model.fit(train_generator, epochs=epochs, validation_data=val_generator)

# ------------------ SAVE MODEL ------------------ #
model_path = os.path.join(save_dir, "blood_model.keras")
model.save(model_path)
print(f"✅ Model saved successfully to: {model_path}")
