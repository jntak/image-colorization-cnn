import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import lab2rgb, rgb2lab, gray2rgb
from skimage.transform import resize
import time



st.set_page_config(page_title="Image Colorization", page_icon="🖼️")
st.title("Image Colorization using Deep Learning 🖼️📸\n")
st.write("Ready to see some magic? Upload a grayscale landscape image to colorize it.")

# Load the encoder (vgg feature extractor layers) and decoder models
encoder_model = tf.keras.models.load_model("./results/model/encoder_model.model")
decoder_model = tf.keras.models.load_model("./results/model/autoencoder_colorize2000.model")


# Create a function to load an image and resize it to be able to be used with our model
def load_and_prep_image(file, img_shape=224):
    """This function loads and prepare a custom grayscale image for prediction

    Args:
        file (path): path to image
        img_shape (int, optional): image shape. Defaults to 224.

    Returns:
        L: L channel of an image
    """
    
    # Read in the image
    img = img_to_array(load_img(file))
    # Resize the image
    img = resize(img, (img_shape, img_shape), anti_aliasing=True)
    img*= 1.0/255
    lab = rgb2lab(img)
    K = lab[:,:,0]
    L = gray2rgb(K)
    L = L.reshape((1,224,224,3))

    return L, K

# A function to help predict(colorize the image)
def predicting(image, decoder_model, encoder_model=encoder_model):

    """ A function to helps predict (colorize) an image

    Args:
        image (file): image file
        decoder_model (keras model): decoder model
        encoder_model (keras model, optional): vgg16 feature extraction layer. Defaults to encoder_model.

    Returns:
        colored: colorized images
    """

    L, K = load_and_prep_image(image)

    vggpred = encoder_model.predict(L, verbose=0)
    ab = decoder_model.predict(vggpred, verbose=0)
    
    ab = ab*128
    color = np.zeros((224, 224, 3))
    color[:,:,0] = K
    color[:,:,1:] = ab
    color = lab2rgb(color)
    color *= 255
    color = color.astype(np.uint8)

    return color



file = st.file_uploader(
    label = "Upload a (B&W) image.\n",
    type = ["jpg", "jpeg", "png"]
)

if not file:
    st.warning("\nPlease upload an image!!")
    st.stop()
else:
    image = file.read()
    st.image(image, use_column_width=True)
    pred_button = st.button("Colorize")


if pred_button:
    st.write("Please Wait!!!!")
    time.sleep(5.0)
    prediction = predicting(file, decoder_model)
    st.image(prediction, caption="Colorized Image", use_column_width=True)
    st.info(f"I hope you liked the painting 😁")
