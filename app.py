import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import pickle

st.set_page_config(page_title="Handwritten Digit Recognizer", layout="centered")

from function import relu, softmax, forward_pass

@st.cache_resource
def load_model_weights(weights_file='weights.pkl'):
    """Loads the pre-trained weights and biases from a pickle file."""
    try:
        with open(weights_file, 'rb') as handle:
            b = pickle.load(handle, encoding="latin1")
        return b[0], b[1], b[2], b[3]
    except FileNotFoundError:
        st.error(f"Error: '{weights_file}' not found. Please ensure the weights file is in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        st.stop()

weight1, bias1, weight2, bias2 = load_model_weights()

st.title("✍️ Handwritten Digit Recognizer")
st.markdown("Draw a single digit (0-9) in the canvas below, then click 'Recognize Digit'!")

if 'canvas_key' not in st.session_state:
    st.session_state.canvas_key = 0
if 'stroke_color' not in st.session_state:
    st.session_state.stroke_color = "#000000" 

canvas_width = 280
canvas_height = 280
stroke_width = 15 
bg_color = "#FFFFFF" 


col1, col2 = st.columns([2, 1])

with col1:
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  
        stroke_width=stroke_width,
        stroke_color=st.session_state.stroke_color,
        background_color=bg_color,
        height=canvas_height,
        width=canvas_width,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}", 
        display_toolbar=True 
    )

with col2:
    st.markdown("### Controls")
    selected_color = st.color_picker("Choose drawing color", st.session_state.stroke_color)
    if selected_color != st.session_state.stroke_color:
        st.session_state.stroke_color = selected_color
        st.rerun() 

    st.markdown("---")

    if st.button("✨ Clear Canvas"):
        st.session_state.canvas_key += 1 
        st.session_state.prediction = None 
        st.rerun() 

    if st.button("🔮 Recognize Digit", type="primary"):
        if canvas_result.image_data is not None:
            img_rgba = Image.fromarray(canvas_result.image_data)
            img_gray = img_rgba.convert('L')

            img_resized = img_gray.resize((28, 28))

            img_array = np.array(img_resized).astype('float32')
            img_array = (255 - img_array) / 255.0

            img_for_prediction = img_array.reshape((1, 28, 28, 1))

            predictions = forward_pass(img_for_prediction, weight1, bias1, weight2, bias2)
            predicted_class = np.argmax(predictions)
            st.session_state.prediction = int(predicted_class) 
        else:
            st.warning("Please draw a digit on the canvas first!")

    st.markdown("---")
    st.markdown("### Prediction")
    
    if 'prediction' in st.session_state and st.session_state.prediction is not None:
        st.success(f"The predicted digit is: **{st.session_state.prediction}**")
    else:
        st.info("Draw a digit and click 'Recognize Digit' to see the prediction.")


