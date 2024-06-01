import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers

# указываем параметры
batch_size = 32
img_size = 256
val_split = 0.2
# путь к датасету
Way_Dataset = r'New_Dataset'

train_datagen = ImageDataGenerator(
    rescale=1 / 255.0,
    validation_split=val_split,
)
val_datagen = ImageDataGenerator(
    rescale=1 / 255.0,
    validation_split=val_split,
)
# Считываем датасет, указываем подмножество training
train_generator = train_datagen.flow_from_directory(
    Way_Dataset,
    target_size=(img_size, img_size),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True,
    subset='training'
)
# Считываем датасет, указываем подмножество validation
val_generator = val_datagen.flow_from_directory(
    Way_Dataset,
    target_size=(img_size, img_size),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True,
    subset='validation'
)

# архитектура нейронной сети
model = tf.keras.models.Sequential([
    Conv2D(16, (5, 5), activation='relu', input_shape=(256, 256, 1)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.5),
    Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.5),
    Dense(33, activation='softmax')
])
# вывод параметров
model.summary()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=['accuracy'])

epochs = 40
# чекпоинты и ранняя остановка
my_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath=f'model.keras', monitor="val_loss", save_best_only=True, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode='min', patience=10, verbose=1)
]
# обучение
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=my_callbacks
)


# Вывод графиков
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epochs)
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
