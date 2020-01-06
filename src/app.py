import streamlit as st
from text2image_model import TextToImageModel


def run_app():

    # initialize the model, takes some time
    model = TextToImageModel()

    # headers
    st.title('Multimodal text2image exercise')
    st.write('by Yury Kashnitsky')

    # get user input from a text area in a Streamlit app
    inserted_text = st.text_area('Insert some text...')
    last_inserted_line = inserted_text.split('\n')[-1]

    # visualize images for the inserted text
    model.play_streamlit(last_inserted_line)


if __name__ == '__main__':
    run_app()

