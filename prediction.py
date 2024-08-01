import numpy as np
import tensorflow as tf

# Loads the new image, and sets it to the proper size
test_image = tf.keras.utils.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64,64))
# Converts PIL object to NumPy Array because Predict function requires Array
test_image = tf.keras.utils.img_to_array(test_image)
# Makes NumPy Array 2D, since we did batched training
test_image = np.expand_dims(test_image, axis=0)

# Loads the previously trained model
loaded_model = tf.keras.models.load_model('catsVsDogs.keras')
# Prediction
result = loaded_model.predict(test_image)

# Prints out the prediction
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)