import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import requests
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO

# Step 1: Build VGG16 pretrained model
def build_model():
    # Load the VGG16 model pre-trained on ImageNet
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze the convolutional layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(2, activation='softmax')(x)  # 2 classes: mask/no mask
    
    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

# Data preprocessing and augmentation
def create_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    valid_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        'Face-Mask-Detection-/facemask/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    
    validation_generator = valid_datagen.flow_from_directory(
        'Face-Mask-Detection-/facemask/valid',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    
    return train_generator, validation_generator

# Training function
def train_model(model, train_generator, validation_generator):
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    # Model checkpoint callback
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        epochs=20,
        validation_data=validation_generator,
        callbacks=[early_stopping, checkpoint]
    )
    
    return history

# Function to preprocess and predict single image
def predict_image(url, model):
    try:
    # Download and open image
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    
    # Display the image
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
    # Preprocess the image
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction = model.predict(img_array)
    class_names = ['with_mask', 'without_mask']
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    
        return predicted_class, confidence
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None

# Main execution
if __name__ == "__main__":
    # Clone repository if not exists
    import os
    import subprocess
    
    if not os.path.exists('Face-Mask-Detection-'):
        try:
            subprocess.run(['git', 'clone', 'https://github.com/chauhanarpit09/Face-Mask-Detection-.git'], 
                         check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error cloning repository: {e}")
            exit(1)
    
    # Build and train model
    model = build_model()
    train_generator, validation_generator = create_data_generators()
    history = train_model(model, train_generator, validation_generator)
    
    # Test with user input
    image_url = input("Enter image URL: ")
    predicted_class, confidence = predict_image(image_url, model)
