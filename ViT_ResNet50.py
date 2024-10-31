from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.applications.resnet50 import ResNet50
import matplotlib.pyplot as plt


def Resnet50Model():
    # Set the path to your image folder
    folder_path = 'D:/FinalProjectViT/20230801_140001'

    # List all files
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):  # Check the file format
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert('RGB')  # 'L' mode means grayscale
            images.append(img)

    # Convert images to array and resize
    image_size = 224
    processed_images = np.array([img_to_array(img.resize((image_size, image_size))) for img in images])

    # Normalize the image data
    processed_images = processed_images / 255.0

    # Convert weight array to a NumPy array
    weights = [np.random.randint(0, 100) for i in range(processed_images.shape[0])] #要改成实际输入的weight
    weights = np.array(weights)

    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(processed_images, weights, test_size=0.2, random_state=42)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)


    # Define the hyperparameters
    batch_size = 32 # You can change this according to your memory
    epochs = 9 # You can change this according to your desired training time
    learning_rate = 0.001 # You can change this according to your optimization strategy

    #-----------------------------------------------------------------------------------
    # Load the ResNet50 model with pretrained weights
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add a custom regression layer on top of the base model
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(1, activation='linear')) # The output layer for regression

    # Compile the model with a mean squared error loss and an Adam optimizer
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

    # Train the model using the generators
    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=epochs, # The number of epochs
        batch_size=batch_size, # The batch size
        validation_split = 0.1
    )

    # test_loss, test_mae = model.evaluate(x_test, y_test)

    # Plot the graph of the loss after nine epochs
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss vs. epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


    plt.plot(history.history['mean_absolute_error'], label='train_mae', color='blue')
    plt.plot(history.history['val_mean_absolute_error'], label='test_mae', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    Resnet50Model()