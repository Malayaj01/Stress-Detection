import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

import os

print("Done ")


train_data_dir = 'data/train/'
val_data_dir = 'data/test/'

train_datagen = ImageDataGenerator(
                 rescale=1./255,
                 rotation_range=30,
                 shear_range=0.3,
                 zoom_range=0.3,
                 horizontal_flip=True,
                 fill_mode='nearest'
                 )

val_datagen = ImageDataGenerator(rescale=1./255 )

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    shuffle=True   
)

validation_generator = val_datagen.flow_from_directory(
    val_data_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)


class_labels = ['Angry','disgust','fearful','happy','neutral','sad','suprised']
img, label = train_generator.__next__()

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1 )))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))


model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))


model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())


num_train_images = 0
for root, dirs, files in os.walk(train_data_dir):
    num_train_images += len(files) 

num_test_images = 0
for root ,dirs, files in os.walk(val_data_dir):
    num_test_images += len(files)

epochs = 100

#early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(train_generator,
                    steps_per_epoch = num_train_images//32,
                    epochs= epochs,
                    validation_data = validation_generator,
                    validation_steps =  num_test_images//32,
                    #callbacks=[early_stopping]
        )

model.save('model_file.keras')

