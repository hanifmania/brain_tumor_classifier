# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN or classifier
classifier = Sequential()

# Step 1 - Convolution, pooling, repeat!
# The convolution is done with 32 filters with size 3 x 3
# The input image is resized into 32 x 32 x 3
# ReLU is used as the activation function after the convolution layer
# The max pooling layer use a kernel size of 2 x 2
classifier.add(Conv2D(32, (3, 3), input_shape = (32, 32, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 2 - Convolution, pooling, repeat!
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection (First and second hidden layer use 64 dan 32 neurons respectively)
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dense(units = 32, activation = 'relu'))
classifier.add(Dense(units = 3, activation = 'softmax'))

# Compiling the CNN or classifier
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2)
                                   # horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training',
                                                 target_size = (32, 32),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

validation_set = test_datagen.flow_from_directory('dataset/validation',
                                            target_size = (32, 32),
                                            batch_size = 32,
                                            class_mode = 'categorical')

classifier.fit_generator(training_set,
                         steps_per_epoch = 500,
                         epochs = 10,
                         validation_data = validation_set,
                         validation_steps = 200)
