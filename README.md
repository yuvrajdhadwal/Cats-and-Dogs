# Cats-and-Dogs
The purpose of this project is to experiment and learn how to use Convolutional Neural Networks (CNN) for computer vision and classification problems that will be very useful for later projects.

The file `model_creation.py` builds a Tensorflow CNN model that can predict whether there is a cat or dog in an image with 80.4% accuracy.

The file `prediction.py` loads in the model:  `catsVsDogs.keras`, and loads an example image to test if the model can predict accurately. Feel free to upload your own image and test if the model can predict!

Training/Test Sets have not been uploaded to GitHub, however this model was trained on 8000 images of dogs and cats and tested with 2000 images of dogs and cats.

Priority was placed on learning how CNNs work, and not building most accurate model. However, we can improve model accuracy by increasing the size of images from 64 x 64 px to anything larger. This was not done, because I wanted swift training as it was more important to learn how to build CNNs rather than make the most accurate one. Check out my Pointing Motion Recognition project if you want to see an accurate model.
