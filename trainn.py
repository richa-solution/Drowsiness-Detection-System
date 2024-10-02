from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Define the model
mymodel = Sequential()
mymodel.add(Conv2D(32, (3, 3), activation='relu', input_shape=(24, 24, 1)))
mymodel.add(MaxPooling2D(pool_size=(2, 2)))
mymodel.add(Dropout(0.25))
mymodel.add(Conv2D(64, (3, 3), activation='relu'))
mymodel.add(MaxPooling2D(pool_size=(2, 2)))
mymodel.add(Dropout(0.25))
mymodel.add(Flatten())
mymodel.add(Dense(128, activation='relu'))
mymodel.add(Dropout(0.5))
mymodel.add(Dense(1, activation='sigmoid'))
mymodel.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
# Define the Data
train = ImageDataGenerator(rescale=1./255, rotation_range=10, zoom_range=0.2, horizontal_flip=True, validation_split=0.2)
test = ImageDataGenerator(rescale=1./255)
train_img = train.flow_from_directory('train', target_size=(24,24), color_mode='grayscale', batch_size=32, class_mode='binary')
test_img = test.flow_from_directory('test', target_size=(24,24), color_mode='grayscale', batch_size=32, class_mode='binary')
# Train and Test the Model
drowsiness_model = mymodel.fit(train_img, epochs=10, validation_data=test_img)
# Save the model in your directory
mymodel.save('drowsiness.h5', drowsiness_model)


