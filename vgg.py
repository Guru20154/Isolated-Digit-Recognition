import os
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def main():
    train_dir = 'image_train_spectrograms'
    test_dir = 'image_test_spectrograms'
    input_size = (224, 224)
    batch_size = 32
    num_classes = 10
    num_epochs = 10
    
    # Data preprocessing and augmentation for training set
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    
    # Data preprocessing for test set (only rescaling)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create train generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=input_size,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    # Create test generator
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=input_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Load the VGG16 model without the top classification layer
    base_model = VGG16(weights='imagenet', include_top=False,
                       input_shape=(input_size[0], input_size[1], 3))
    
    # Freeze the layers of the pre-trained model
    for layer in base_model.layers:
        layer.trainable = False
    
    # Create the model
    model = tf.keras.models.Sequential()
    model.add(base_model)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train the model
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=num_epochs,
        validation_data=test_generator,
        validation_steps=test_generator.samples // batch_size
    )
    
    # Save the trained model
    model.save('vgg_model.h5')

if __name__ == '__main__':
    main()
