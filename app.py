import streamlit as st
import tensorflow as tf
import numpy as np
import gdown

# Function to download a file using gdown
def download_file_from_google_drive(url, output):
    try:
        gdown.download(url, output, quiet=False)
    except Exception as e:
        st.error(f"Failed to download the file from Google Drive: {str(e)}")

# Download the model weights
url_weights = "https://drive.google.com/uc?id=171x-xwdqm2dXMOyZiCfPwY6xU6IWFGyF"
output_weights = "model.weights.h5"
download_file_from_google_drive(url_weights, output_weights)

# Load the weights into the model
model.load_weights(output_weights)

# TensorFlow model prediction function
def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select", ["Home", "About", "Disease Recognition"])

#HOME PAGE
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION AI")
    image_path = "https://www.google.com/url?sa=i&url=https%3A%2F%2Fdepositphotos.com%2Fphotos%2Fplant.html&psig=AOvVaw3S2S-OhYERARbmg9cp9rzS&ust=1724320278447000&source=images&cd=vfe&opi=89978449&ved=0CBEQjRxqFwoTCOietuHnhYgDFQAAAAAdAAAAABAE"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """
    )

#ABOUT PAGE
elif(app_mode=='About'):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """
    )    

#PREDICTION PAGE
elif(app_mode=='Disease Recognition'):
    st.header("Disease Recognition")  
    test_image = st.file_uploader("Choose an image")
    if(st.button("Show Image")):
        st.image(test_image, use_column_width=True)  
    if(st.button("Predict")):
        st.balloons()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Define class
        class_name = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']
        st.success("Model predicts that, it's a {}".format(class_name[result_index]))
