import numpy as np
import tensorflow as tf
# from tensorflow.keras.preprocessing import image

test_image = tf.keras.utils.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64,64))
test_image = tf.keras.utils.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

loaded_model = tf.keras.models.load_model('catsVsDogs.keras')
result = loaded_model.predict(test_image)

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)