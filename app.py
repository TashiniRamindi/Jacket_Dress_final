import streamlit as st
import joblib
import pandas as pd
import base64

# Function to load and encode the background image
def set_background_image(image_file):
    with open(image_file, "rb") as img:
        base64_str = base64.b64encode(img.read()).decode()
    background_css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{base64_str}");
        background-size: cover;
    }}
    </style>
    """
    st.markdown(background_css, unsafe_allow_html=True)

# Function to load and display a top image with increased size
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return encoded

def set_image_top(image_path):
    base64_str = get_base64_image(image_path)
    st.markdown(f'<img src="data:image/jpeg;base64,{base64_str}" style="display:block;margin-left:auto;margin-right:auto;width:80%;">', unsafe_allow_html=True)

# Set the background image
set_background_image("blue.jpg")  # Background image

# Set an image at the top with increased size
set_image_top("background.jpg")  # Top image file

# Load the saved models for both jacket and dress
model_dress = joblib.load("classification_model_dress.pkl")
model_jacket = joblib.load("classification_model_jacket.pkl")

columns_dress = joblib.load("dress_X_train.pkl")
columns_jacket = joblib.load("jacket_X_train.pkl")

import streamlit as st
import pandas as pd
import joblib

# Load the saved model and feature columns
model = joblib.load("classification_model_jacket.pkl")
feature_columns = joblib.load("jacket_X_train.pkl")

# Create a function to transform user inputs into the expected format for the model
def preprocess_input(user_input):
    # Mapping categorical input values to numerical values
    fit_mapping = {'slim_fit': 0, 'regular_fit': 1, 'relaxed_fit': 2, 'oversize_fit': 3}
    length_mapping = {'short': 0, 'medium': 1, 'long': 2}
    sleeve_length_mapping = {'sleeveless': 0, 'elbow_length': 1, 'long_sleeve': 2}
    season_mapping = {'spring': 0, 'summer': 1, 'winter': 2, 'autumn': 3}

    # Map categorical values to the numerical ones expected by the model
    user_input['Fit'] = fit_mapping[user_input['Fit']]
    user_input['Length'] = length_mapping[user_input['Length']]
    user_input['Sleeve Length'] = sleeve_length_mapping[user_input['Sleeve Length']]
    user_input['Breathable'] = 1 if user_input['Breathable'] == 'Yes' else 0
    user_input['Lightweight'] = 1 if user_input['Lightweight'] == 'Yes' else 0
    user_input['Water_Repellent'] = 1 if user_input['Water_Repellent'] == 'Yes' else 0

    # Convert categorical columns into one-hot encoded features
    categorical_columns = ['Outerwear Type', 'Hemline', 'Material', 'Neckline', 'Collar', 'Product Colour', 
                           'Sleeve Style', 'Pattern', 'Style']
    
    # Create DataFrame from the user input
    input_data = pd.DataFrame([user_input])
    
    # Perform one-hot encoding
    input_data_dummies = pd.get_dummies(input_data[categorical_columns], drop_first=True)
    input_data = pd.concat([input_data, input_data_dummies], axis=1)
    input_data = input_data.drop(columns=categorical_columns)
    
    # Ensure the input data matches the model's input columns
    input_data = input_data.reindex(columns=feature_columns, fill_value=0)
    
    return input_data

# Function to predict the season
def predict_season(user_input):
    # Preprocess the input
    input_data = preprocess_input(user_input)
    
    # Make prediction using the model
    prediction = model.predict(input_data)
    
    # Map the prediction back to season
    season_mapping_inv = {0: 'spring', 1: 'summer', 2: 'winter', 3: 'autumn'}
    predicted_season = season_mapping_inv[prediction[0]]
    
    return predicted_season

# Streamlit user interface (UI) for input
st.title("Jacket Season Prediction")

# Get user input using Streamlit widgets
user_input = {
    import streamlit as st

# User input for Outerwear Type
cloth_type = st.selectbox('What type of outerwear is this?', ['jacket', 'vest', 'coat'], index=None)

# Define options based on the selected outerwear type
if cloth_type == 'jacket':
    # Display options for Jacket Style if the outerwear type is 'jacket'
    style = st.selectbox('What style is the jacket?', ['bomber', 'trucker', 'windbreaker', 'soft_shell', 'sweatshirt', 
                                                     'other_style', 'harrington', 'rain_jacket', 'cargo', 'shirt', 
                                                     'blazer', 'anorak', 'hardshell'])
elif cloth_type == 'coat':
    # Display options for Coat Style if the outerwear type is 'coat'
    style = st.selectbox('What style is the coat?', ['puffer', 'other_style', 'parka', 'trench', 'cocoon', 'overcoat', 
                                                   'peacoat', 'barn'])
else:
    # Display options for Vest Style if the outerwear type is 'vest'
    style = st.selectbox('What style is the vest?', ['gilet', 'puffer', 'trucker', 'other_style', 'suit'])

    'Fit': st.selectbox('What is the fit of the jacket?', ['regular_fit', 'relaxed_fit', 'slim_fit', 'oversize_fit'], index=None),
    'Length': st.selectbox('What is the length of the jacket?', ['short', 'medium', 'long'], index=None),
    'Sleeve Length': st.selectbox('What sleeve length does the jacket have?', ['long_sleeve', 'sleeveless', 'elbow_length'], index=None),
    'Collar': st.selectbox('What type of collar does the jacket have?', ['point', 'no collar', 'band', 'notch', 'lapel','other_collar'], index=None),
    'Neckline': st.selectbox('What type of neckline does the jacket have?', ['collared_neck', 'hooded', 'funnel_neck', 'v_neck', 'other_neck'], index=None),
    'Hemline': st.selectbox('What type of hemline does the jacket have?', ['ribbed_hem', 'straight_hem', 'other_hem'], index=None),
    'Sleeve Style': st.selectbox('What sleeve style does the jacket have?', ['cuff_sleeve', 'no_sleeve', 'plain_sleeve', 'raglan_sleeve','other_sleeve_style'], index=None),
    'Pattern': st.selectbox('What pattern does the jacket have?', ['solid_or_plain', 'multicolor', 'printed','plaid', 'cable_knit', 'tie_dry', 'houndstooth', 'chevron','other'], index=None),
    'Product Colour': st.selectbox('What color is the jacket?', ['black', 'grey', 'blue', 'red', 'white', 'brown', 'yellow', 'pink', 'green', 'cream', 'beige', 'purple', 'orange', 'multi_color'], index=None),
    'Material': st.selectbox('What material is the jacket made from?', ['Polyamide', 'Cotton', 'Polyester', 'Nylon',  'fleece', 'Wool', 'denim', 'leather', 'faux_fur', 'corduroy', 'rib_knit', 'Other material'], index=None),
    'Breathable': st.radio('Is the jacket breathable?', ('Yes', 'No'), index=None),
    'Lightweight': st.radio('Is the jacket lightweight?', ('Yes', 'No'), index=None),
    'Water_Repellent': st.radio('Is the jacket water repellent?', ('Yes', 'No'), index=None)
}
# Example of displaying the result
st.write(f'Selected Outerwear Type: {cloth_type}')
st.write(f'Selected Style: {style}')

# Predict the season and display the result
if st.button('Predict Season'):
    predicted_season = predict_season(user_input)
    st.write(f"The predicted season for this jacket is: {predicted_season}")
