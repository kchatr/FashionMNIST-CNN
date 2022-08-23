# Libraries for Machine Learning & Data Handling
import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# Helper Libraries
import math
import numpy as np
import matplotlib.pyplot as plt

# Set up logging
import logging
logger = tf.getLogger()
logger.setLevel(logging.ERROR)

def normalize_data(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

def plot_image(index, pred_arr, labels, class_names, images):
    predictions_array, label, img = pred_arr[index], labels[index], images[index] 
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img[..., 0], cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == label:
        color = 'blue'
    else:
        color = 'red'
    
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                            100*np.max(predictions_array),
                            class_names[label]),
                            color=color)

def plot_val_array(index, pred_arr, labels):
    predictions_array, label = pred_arr[index], labels[index]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1]) 
    predicted_label = np.argmax(predictions_array)
    
    thisplot[predicted_label].set_color('red')
    thisplot[label].set_color('blue')


def main():
    # Import & Split Dataset
    dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
    train_ds, test_ds = dataset['train'], dataset['test']
    class_names = metadata.features['label'].names
    num_train_examples = metadata.splits['train'].num_examples
    num_test_examples = metadata.splits['test'].num_examples

    # Data Preprocessing
    # Normalize and Cache data 
    train_ds = train_ds.map(normalize_data)
    test_ds = test_ds.map(normalize_data)
    train_ds = train_ds.cache()
    test_ds = test_ds.cache()

    # Explore Data
    # Print Breakdown of Training Data to Testing Data
    num_train_examples = metadata.splits['train'].num_examples
    num_test_examples = metadata.splits['test'].num_examples
    print("Number of training examples: {}".format(num_train_examples))
    print("Number of test examples:     {}".format(num_test_examples))

    # Display First 25 Training Examples for Verification
    plt.figure(figsize=(15,15))
    for i, (image, label) in enumerate(train_ds.take(25)):
        image = image.numpy().reshape((28,28))
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)
        plt.xlabel(class_names[label])
    plt.show()

    # Initialize the Model
    # Uses two convolution layers followed by two densely connected layers, with the final layer serving as the output layer.
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.keras.activations.relu, input_shape = (28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax)
    ])

    # Compile the Model
    model.compile(optimizer='adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    # Train the Model
    BATCH_SIZE = 32
    train_ds = train_ds.cache().shuffle(num_train_examples).batch(BATCH_SIZE)
    test_ds=test_ds.cache().batch(BATCH_SIZE)

    model.fit(x=train_ds, epochs=5)

    # Evaluate Model Performance on Test Data
    test_loss, test_accuracy = model.evaluate(test_ds, steps=math.ceil(num_test_examples/BATCH_SIZE))
    print('Accuracy on test dataset:', test_accuracy)

    # Generate Predictions on Test Data
    predictions, test_images, test_labels = None, None, None
    for test_images, test_labels in test_ds.take(1):
        test_images = test_images.numpy()
        test_labels = test_labels.numpy()
        predictions = model.predict(test_images)
    
    # Plot Model Predictions
    num_rows = 5
    num_cols = 3
    num_images = num_rows*num_cols
    
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions, test_labels, class_names, test_images)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_val_array(i, predictions, test_labels)
    plt.show()