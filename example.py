import numpy as np
import cv2 as cv
from PIL import Image
from keras.preprocessing import image
from keras.models import load_model


# load images to normalize input data
mean_image = image.img_to_array(image.load_img("mean_image.png",target_size=(49,49)))
std_image = image.img_to_array(image.load_img("std_image.png",target_size=(49,49)))

# load model
model=load_model('irc-cnn-009-0.642313.h5')


def normalize(image,mean_image,std_image):
    return np.divide((image-mean_image),std_image)

def prediction(np_face,mean_image,std_image):
    np_face =np.expand_dims(np_face, axis=0)
    preds= model.predict(normalize(np_face,mean_image,std_image))
    return preds

# load test image
img = cv.imread('angry.jpg',cv.IMREAD_COLOR)
width = np.shape(img)[1]
height = np.shape(img)[0]

# we use the facial detector of opencv
face_cascade = cv.CascadeClassifier('face.xml')
faces = face_cascade.detectMultiScale(img, 1.3, 5)

# we loop through faces
for index,(x,y,w,h) in enumerate(faces):
    np_face = img[y:y+h, x:x+w]
    # converting into PIL.Image object to resize
    pil_face = Image.fromarray(np_face, 'RGB')
    pil_face = pil_face.resize((49, 49), resample=Image.BILINEAR)

    # converting into np.array
    np_face = np.flip(np.asarray(pil_face, dtype=np.uint8),2)

    preds=prediction(np_face,mean_image,std_image)
    print(preds)
