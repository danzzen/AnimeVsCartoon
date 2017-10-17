#lets do
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import Dense

from keras.models import model_from_json
classifier=Sequential()

classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))
####
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datgen=ImageDataGenerator(rescale=1./255,shear_range=0.3,zoom_range=0.2,horizontal_flip=True)

test_datgen=ImageDataGenerator(rescale=1./255)

training_set = train_datgen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datgen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=691,
                         epochs=12)
model_json = classifier.to_json()
with open("classifier.json", "w") as json_file:
    json_file.write(model_json)
classifier.save_weights("classifier.h5")
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/test_set/cartoon.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'anime'
else:
    prediction = 'cartoon'
print(prediction)

