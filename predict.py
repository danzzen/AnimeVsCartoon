from sklearn.externals import joblib
import main
from keras.models import model_from_json

json_file = open('classifier.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)
# load weights into new model
classifier.load_weights("classifier.h5")
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/test_set/images.png', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
main.training_set.class_indices
if result[0][0] == 1:
    prediction = 'anime'
else:
    prediction = 'cartoon'
print(prediction)
