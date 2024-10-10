import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2,InceptionV3,VGG19
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import itertools
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, Model
from tensorflow.keras.models import Sequential
import pathlib
import sys

DATASET_ROOT_PATH = "./Datasets"

# Generate test and train data from the dataset
def generate_train_test_data(train_data_dir,test_data_dir,batch_size,img_size):
    # ImageDataGenerator with resizing
    train_datagen = ImageDataGenerator(
        rescale=1./255,
    )
    test_datagen = ImageDataGenerator(
        rescale=1./255
    )

    # Load and normalize training data
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Load and normalize testing data
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, test_generator

# Plot confusion matrix and classification report
def _plot_confusion_matrix_and_report(y_true, y_pred, class_names):
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

# generate the overall f1/accuracy score
def _overall_metrics(loaded_y_true, loaded_y_pred):
    # Calculate and print overall F1 score for the loaded model
    loaded_overall_f1 = f1_score(loaded_y_true, loaded_y_pred, average='weighted')
    print(f'Overall F1 Score (loaded model): {loaded_overall_f1}')

    # Calculate and print overall accuracy for the loaded model
    loaded_overall_accuracy = accuracy_score(loaded_y_true, loaded_y_pred)
    print(f'Overall Accuracy (loaded model): {loaded_overall_accuracy}')

# generate pre processing image dataset
def train_test_model(train_data_dir,test_data_dir,batch_size,img_height,img_width):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_data_dir,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_data_dir,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    return train_ds, test_ds


# generate report graph
def _report_graph (test_labels, pred_labels, class_names):
    print("\nTest Classification Report:")
    print(classification_report(test_labels, pred_labels, target_names=class_names))

    print("\nTest Confusion Matrix:")
    test_conf_matrix = confusion_matrix(test_labels, pred_labels)
    print(test_conf_matrix)

    # Plotting test confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(test_conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Test Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = test_conf_matrix.max() / 2.
    for i, j in itertools.product(range(test_conf_matrix.shape[0]), range(test_conf_matrix.shape[1])):
        plt.text(j, i, format(test_conf_matrix[i, j], fmt),
                horizontalalignment="center",
                color="white" if test_conf_matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

# generate report
def _report (test_labels, pred_labels):
    # Calculate and print overall F1 score for test data
    test_overall_f1 = f1_score(test_labels, pred_labels, average='weighted')
    print(f'Test Overall F1 Score: {test_overall_f1}')

    # Calculate and print overall accuracy for test data
    test_overall_accuracy = accuracy_score(test_labels, pred_labels)
    print(f'Test Overall Accuracy: {test_overall_accuracy}')



def MobileNet_V2(train_data_dir, test_data_dir, dtype):
    # Define image size and batch size
    batch_size = 32
    img_size = (224, 224)
   
    # Load and normalize training data
    train_generator, test_generator = generate_train_test_data(
        train_data_dir, test_data_dir,
        batch_size, img_size,
    )

    # Load MobileNetV2 base model (pre-trained on ImageNet)
    base_model = MobileNetV2(weights='imagenet', include_top=False)

    # Add custom head on top of MobileNetV2
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)

    # Final model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze base layers during initial training
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Train the model
    model.fit(
        train_generator,
        epochs=10,  # adjust as needed
    )

    # Save the model
    model.save('mobilenet_v2_model{}.keras'.format(dtype))

    # Load the model
    loaded_model = load_model('mobilenet_v2_model{}.keras'.format(dtype))

    # Evaluate the loaded model on the testing set
    loaded_predictions = loaded_model.predict(test_generator)

    # plotting classification report
    loaded_y_pred = tf.argmax(loaded_predictions, axis=1)
    loaded_y_true = test_generator.classes
    class_names = list(test_generator.class_indices.keys())

    _plot_confusion_matrix_and_report(loaded_y_true, loaded_y_pred, class_names)
    _overall_metrics(loaded_y_true, loaded_y_pred)

    # Print model summary
    loaded_model.summary()


def Inception_V3(train_data_dir, test_data_dir, dtype):
    # Define image size and batch size
    batch_size = 32
    img_size = (299, 299)  # InceptionV3 requires input size to be (299, 299)

    # Load and normalize training data
    train_generator, test_generator = generate_train_test_data(
        train_data_dir, test_data_dir,
        batch_size, img_size,
    )

    # Load InceptionV3 base model (pre-trained on ImageNet)
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

    # Add custom head on top of InceptionV3
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)

    # Final model
    model1 = Model(inputs=base_model.input, outputs=predictions)

    # Freeze base layers during initial training
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model1.compile(optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Train the model
    model1.fit(
        train_generator,
        epochs=10,  # adjust as needed
    )

    # Save the model
    model1.save('inception_v3_model{}.keras'.format(dtype))
    # Load the model
    loaded_model = load_model('inception_v3_model{}.keras'.format(dtype))

    # Evaluate the loaded model on the testing set
    loaded_predictions = loaded_model.predict(test_generator)
    loaded_y_pred = tf.argmax(loaded_predictions, axis=1)
    loaded_y_true = test_generator.classes


    class_names = list(test_generator.class_indices.keys())
    _plot_confusion_matrix_and_report(loaded_y_true, loaded_y_pred, class_names)

    _overall_metrics(loaded_y_true, loaded_y_pred)

    loaded_model.summary()


def VGG_19(train_data_dir, test_data_dir, dtype):

    batch_size = 32
    img_size = (224, 224)  

    # Load and normalize training data
    train_generator, test_generator = generate_train_test_data(
            train_data_dir, test_data_dir,
            batch_size, img_size,
        )

    # Load VGG19 base model (pre-trained on ImageNet)
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Add custom head on top of VGG19
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(len(train_generator.class_indices), activation='softmax')(x)

    # Final model
    model2 = Model(inputs=base_model.input, outputs=predictions)

    # Freeze base layers during initial training
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model2.compile(optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Train the model
    model2.fit(
        train_generator,
        epochs=10,  # adjust as needed
    )

    # Save the model
    model2.save('vgg19_model{}.keras'.format(dtype))
    # Load the model
    loaded_model = load_model('vgg19_model{}.keras'.format(dtype))

    # Evaluate the loaded model on the testing set
    loaded_predictions = loaded_model.predict(test_generator)
    loaded_y_pred = tf.argmax(loaded_predictions, axis=1)
    loaded_y_true = test_generator.classes

    class_names = list(test_generator.class_indices.keys())
    _plot_confusion_matrix_and_report(loaded_y_true, loaded_y_pred, class_names)
    _overall_metrics(loaded_y_true, loaded_y_pred)

    loaded_model.summary()



def EfficientNet_V2(train_data_dir, test_data_dir, dtype):

    batch_size = 32
    img_height, img_width = 224, 224

    train_ds, test_ds = train_test_model(train_data_dir,test_data_dir,batch_size,img_height,img_width)

    class_names = train_ds.class_names
    # Define the model architecture
    def create_model():
        inputs = layers.Input(shape=(224, 224, 3))
        pretrained_model = tf.keras.applications.EfficientNetV2B0(
            include_top=False,
            input_tensor=inputs,
            pooling='avg',  # Using average pooling to flatten the feature maps
            weights='imagenet'
        )
        pretrained_model.trainable = False

        x = layers.Dense(512, activation='relu')(pretrained_model.output)
        outputs = layers.Dense(len(class_names), activation='softmax')(x)

        model = tf.keras.Model(inputs, outputs)

        model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    # Create and train a new model
    efficientnet_model = create_model()
    epochs = 10
    efficientnet_model.fit(
        train_ds,
        epochs=epochs
    )

    # Save the model
    efficientnet_model.save('efficientnet_model{}.keras'.format(dtype))

    # Load the saved model
    loaded_model = tf.keras.models.load_model('efficientnet_model{}.keras'.format(dtype))

    loaded_model.summary()

    # Making Predictions and Evaluating Metrics

    test_images = []
    test_labels = []

    for images, labels in test_ds:
        test_images.append(images.numpy())
        test_labels.append(labels.numpy())

    test_images = np.concatenate(test_images)
    test_labels = np.concatenate(test_labels)

    test_predictions = loaded_model.predict(test_images)
    pred_labels = np.argmax(test_predictions, axis=1)

    _report_graph (test_labels, pred_labels, class_names)
    _report (test_labels, pred_labels)




def ResNet_50(train_data_dir, test_data_dir, dtype):
    img_height, img_width = 224, 224
    batch_size = 32
    train_ds, test_ds = train_test_model(train_data_dir,test_data_dir,batch_size,img_height,img_width)

    class_names = train_ds.class_names
    print(class_names)

    # Define the model architecture
    def create_model():
        base_model = tf.keras.applications.ResNet50(
            include_top=False,
            input_shape=(224, 224, 3),
            pooling='avg',
            weights='imagenet'
        )
        for layer in base_model.layers:
            layer.trainable = False

        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = base_model(inputs, training=False)
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu')(x)
        outputs = layers.Dense(len(class_names), activation='softmax')(x)

        model = tf.keras.Model(inputs, outputs)

        model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    # Create and train a new model
    resnet_model = create_model()
    epochs = 10
    resnet_model.fit(
        train_ds,
        epochs=epochs
    )

    # Save the model
    model_path = 'resnet50_model{}.keras'.format(dtype)
    resnet_model.save(model_path)
    print(f'Model saved to {model_path}')

    # Load the model
    loaded_model = tf.keras.models.load_model(model_path)
    print('Model loaded successfully')

    # Making Predictions and Evaluating Metrics
    test_images = []
    test_labels = []

    for images, labels in test_ds:
        test_images.append(images.numpy())
        test_labels.append(labels.numpy())

    test_images = np.concatenate(test_images)
    test_labels = np.concatenate(test_labels)

    predictions = loaded_model.predict(test_images)
    pred_labels = np.argmax(predictions, axis=1)

    loaded_model.summary()

    _report_graph(test_labels, pred_labels, class_names)
    _report(test_labels, pred_labels)