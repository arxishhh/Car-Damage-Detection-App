import streamlit as st
from model_helper import predict
st.title('Vehicle Damage Detection')

uploaded_file = st.file_uploader('Upload the File',type = ['jpg','png','jpeg'])

if uploaded_file:
    image_path = 'temp_file.jpg'
    with open(image_path,'wb') as f:
        f.write(uploaded_file.getbuffer())
        predict = predict(image_path)
    st.image(uploaded_file,caption = 'Uploaded Vehicle Image',use_container_width=True)
    idx = predict.index('_')
    category = predict[idx+1:]
    if category == 'Breakage' or category == 'Crushed':
        st.info('Vehicle Damage Detected')
    else :
        st.info('Vehicle is Fine')