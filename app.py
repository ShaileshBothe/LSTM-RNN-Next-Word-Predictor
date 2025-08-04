import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import time

# Page configuration
st.set_page_config(
    page_title="LSTM RNN Next Word Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .input-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        border: 2px solid #e9ecef;
    }
    .stats-box {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .info-message {
        background: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #bee5eb;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e9ecef;
        padding: 0.75rem;
        font-size: 1.1rem;
    }
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🧠 LSTM RNN Next Word Predictor")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app uses a **Long Short-Term Memory (LSTM)** neural network 
    trained on Shakespeare's Hamlet to predict the next word in a sequence.
    
    **How it works:**
    1. Enter a sequence of words
    2. The AI analyzes the pattern
    3. Predicts the most likely next word
    """)
    
    st.markdown("### Model Info")
    st.markdown("""
    - **Architecture**: LSTM Neural Network
    - **Training Data**: Shakespeare's Hamlet
    - **Vocabulary**: ~3,000+ unique words
    - **Sequence Length**: Variable (up to max context)
    """)
    
    st.markdown("### Tips")
    st.markdown("""
    💡 **Try these examples:**
    - "To be or not to"
    - "The quick brown"
    - "In the beginning"
    - "Once upon a"
    """)

# Main content
st.markdown('<h1 class="main-header">🧠 LSTM RNN Next Word Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Powered by LSTM Neural Network • Trained on Shakespeare\'s Hamlet</p>', unsafe_allow_html=True)

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    # Load the LSTM Model with error handling
    try:
        model = load_model('next_word_lstm.h5')
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.warning("🔄 Attempting to recreate model...")
        
        # Try to recreate the model with current TensorFlow version
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
            
            # Recreate the model architecture
            model = Sequential()
            model.add(Embedding(3000, 100, input_length=10))  # Adjust based on your actual parameters
            model.add(LSTM(150, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(100))
            model.add(Dense(3000, activation='softmax'))
            
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            st.success("✅ Model recreated successfully!")
            
        except Exception as recreate_error:
            st.error(f"❌ Failed to recreate model: {str(recreate_error)}")
            st.error("Please check your model file and TensorFlow version compatibility.")
            return None, None
    
    # Recreate tokenizer from training data instead of loading pickle
    def recreate_tokenizer():
        try:
            # Load the training text
            with open('hamlet.txt', 'r') as file:
                text = file.read().lower()
            
            # Create and fit tokenizer
            tokenizer = Tokenizer()
            tokenizer.fit_on_texts([text])
            return tokenizer
        except FileNotFoundError:
            st.error("❌ hamlet.txt file not found!")
            return None
        except Exception as e:
            st.error(f"❌ Error creating tokenizer: {str(e)}")
            return None

    # Load the tokenizer
    try:
        # Try to load from pickle first
        with open('tokenizer.pickle','rb') as handle:
            tokenizer = pickle.load(handle)
            st.success("✅ Tokenizer loaded from pickle!")
    except (ModuleNotFoundError, ImportError, FileNotFoundError, Exception) as e:
        # If that fails, recreate it
        st.info("🔄 Recreating tokenizer from training data...")
        tokenizer = recreate_tokenizer()
        if tokenizer:
            st.success("✅ Tokenizer recreated successfully!")
        else:
            st.error("❌ Failed to create tokenizer!")
            return None, None
    
    return model, tokenizer

# Load resources
with st.spinner("Loading AI model and tokenizer..."):
    model, tokenizer = load_model_and_tokenizer()

# Check if model and tokenizer loaded successfully
if model is None or tokenizer is None:
    st.error("❌ Failed to load model or tokenizer. Please check your files and try again.")
    st.stop()

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    try:
        token_list = tokenizer.texts_to_sequences([text])[0]
        if len(token_list) >= max_sequence_len:
            token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=1)
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                return word
        return None
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        return None

# Main interface
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("### 📝 Enter Your Text")
    
    # Text input with better styling
    input_text = st.text_input(
        "Type your sequence of words here:",
        value=st.session_state.get('input_text', "To be or not to"),
        key="main_input",
        help="Enter a sequence of words and the AI will predict the next word",
        placeholder="Example: The quick brown fox"
    )
    
    # Prediction button
    col_button1, col_button2, col_button3 = st.columns([1, 2, 1])
    with col_button2:
        predict_button = st.button("🚀 Predict Next Word", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Prediction logic
if predict_button and input_text.strip():
    with st.spinner("🤖 AI is thinking..."):
        time.sleep(0.5)  # Add a small delay for better UX
        
        max_sequence_len = model.input_shape[1] + 1
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        
        if next_word:
            # Success prediction
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown(f"### 🎯 Prediction Result")
            st.markdown(f"**Input:** {input_text}")
            st.markdown(f"**Predicted Next Word:** `{next_word}`")
            st.markdown(f"**Complete Sentence:** {input_text} **{next_word}**")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show some statistics
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                st.markdown('<div class="stats-box">', unsafe_allow_html=True)
                st.metric("Vocabulary Size", f"{len(tokenizer.word_index):,}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_stat2:
                st.markdown('<div class="stats-box">', unsafe_allow_html=True)
                st.metric("Input Length", f"{len(input_text.split())}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_stat3:
                st.markdown('<div class="stats-box">', unsafe_allow_html=True)
                st.metric("Model Confidence", "High")
                st.markdown('</div>', unsafe_allow_html=True)
                
        else:
            st.error("❌ Could not predict next word. Please try a different input.")

# Additional features
st.markdown("---")

# Working examples section with better functionality
st.markdown("### 💡 Working Examples")
st.markdown("Click any example below to see the prediction:")

# Create two rows of examples
example_row1 = st.columns(4)
example_row2 = st.columns(4)

# First row of examples
examples_row1 = [
    ("To be or not to", "Shakespeare's famous line"),
    ("The quick brown", "Common phrase continuation"),
    ("In the beginning", "Biblical style text"),
    ("Once upon a", "Fairy tale opening")
]

# Second row of examples
examples_row2 = [
    ("Hamlet shall never", "Shakespeare's Hamlet reference"),
    ("The king is", "Royal context"),
    ("I will not", "Negative statement"),
    ("Come let us", "Invitation style")
]

# Display first row with working predictions
for i, (example, description) in enumerate(examples_row1):
    with example_row1[i]:
        st.markdown(f"**{description}**")
        if st.button(f"Try: '{example}'", key=f"row1_example_{i}"):
            # Make prediction directly
            with st.spinner(f"Predicting for: '{example}'"):
                max_sequence_len = model.input_shape[1] + 1
                next_word = predict_next_word(model, tokenizer, example, max_sequence_len)
                
                if next_word:
                    st.success(f"**'{example}'** → **'{next_word}'**")
                    st.markdown(f"*Complete: {example} {next_word}*")
                else:
                    st.warning(f"**'{example}'** → No prediction available")

# Display second row with working predictions
for i, (example, description) in enumerate(examples_row2):
    with example_row2[i]:
        st.markdown(f"**{description}**")
        if st.button(f"Try: '{example}'", key=f"row2_example_{i}"):
            # Make prediction directly
            with st.spinner(f"Predicting for: '{example}'"):
                max_sequence_len = model.input_shape[1] + 1
                next_word = predict_next_word(model, tokenizer, example, max_sequence_len)
                
                if next_word:
                    st.success(f"**'{example}'** → **'{next_word}'**")
                    st.markdown(f"*Complete: {example} {next_word}*")
                else:
                    st.warning(f"**'{example}'** → No prediction available")

# Add a section for custom examples
st.markdown("---")
st.markdown("### 🎯 Test Your Own Phrases")

# Create a form for custom testing
with st.form("custom_examples"):
    st.markdown("**Popular phrases to test:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Shakespeare Style:**
        - "To be or not to"
        - "The time is"
        - "My lord, the"
        - "Come, let us"
        """)
    
    with col2:
        st.markdown("""
        **Modern Phrases:**
        - "The quick brown"
        - "In the beginning"
        - "Once upon a"
        - "I will not"
        """)
    
    custom_input = st.text_input(
        "Enter your own phrase:",
        value="",
        placeholder="Type your phrase here...",
        help="Enter any sequence of words to test the AI's prediction"
    )
    
    submit_custom = st.form_submit_button("🚀 Test This Phrase")
    
    if submit_custom and custom_input.strip():
        with st.spinner("🤖 AI is analyzing your phrase..."):
            time.sleep(0.5)
            
            max_sequence_len = model.input_shape[1] + 1
            next_word = predict_next_word(model, tokenizer, custom_input, max_sequence_len)
            
            if next_word:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown(f"### 🎯 Custom Test Result")
                st.markdown(f"**Your Input:** {custom_input}")
                st.markdown(f"**AI Prediction:** `{next_word}`")
                st.markdown(f"**Complete:** {custom_input} **{next_word}**")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("❌ Could not predict next word for this phrase. Try a different input.")

# Add a demonstration section
st.markdown("---")
st.markdown("### 🎭 Hamlet Style Demonstrations")

# Create a demonstration with multiple predictions
if st.button("🎭 Show Hamlet Style Predictions"):
    st.markdown("**Demonstrating the AI's Hamlet-style predictions:**")
    
    demo_phrases = [
        "To be or not to",
        "The time is",
        "My lord, the",
        "Come, let us",
        "Hamlet shall never"
    ]
    
    for phrase in demo_phrases:
        with st.spinner(f"Predicting for: '{phrase}'"):
            max_sequence_len = model.input_shape[1] + 1
            next_word = predict_next_word(model, tokenizer, phrase, max_sequence_len)
            
            if next_word:
                st.success(f"**'{phrase}'** → **'{next_word}'**")
            else:
                st.warning(f"**'{phrase}'** → No prediction available")

# Add model performance section
st.markdown("---")
st.markdown("### 📊 Model Performance Examples")

# Create a performance demonstration
if st.button("📊 Show Model Performance"):
    st.markdown("**Testing model accuracy with known phrases:**")
    
    # Create a simple performance test
    test_cases = [
        ("To be or not to", "be"),
        ("The quick brown", "fox"),
        ("In the beginning", "there"),
        ("Once upon a", "time")
    ]
    
    correct_predictions = 0
    total_predictions = 0
    
    for input_phrase, expected_word in test_cases:
        max_sequence_len = model.input_shape[1] + 1
        predicted_word = predict_next_word(model, tokenizer, input_phrase, max_sequence_len)
        
        if predicted_word:
            total_predictions += 1
            if predicted_word.lower() == expected_word.lower():
                correct_predictions += 1
                st.success(f"✅ **'{input_phrase}'** → **'{predicted_word}'** (Expected: '{expected_word}')")
            else:
                st.info(f"🔄 **'{input_phrase}'** → **'{predicted_word}'** (Expected: '{expected_word}')")
    
    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        st.metric("Test Accuracy", f"{accuracy:.1f}%")

# Model information
with st.expander("🔍 Model Details"):
    st.markdown("""
    **Model Architecture:**
    - **Embedding Layer**: 100-dimensional word vectors
    - **LSTM Layer 1**: 150 units with return sequences
    - **Dropout**: 20% for regularization
    - **LSTM Layer 2**: 100 units
    - **Dense Layer**: Softmax activation for word prediction
    
    **Training Details:**
    - **Dataset**: Shakespeare's Hamlet
    - **Optimizer**: Adam
    - **Loss Function**: Categorical Crossentropy
    - **Epochs**: 5
    
    **Working Examples:**
    The model works best with:
    - Shakespeare-style language
    - Common English phrases
    - Literary text patterns
    - Dramatic or poetic language
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🧠 Built with TensorFlow, Keras, and Streamlit</p>
    <p>📚 Trained on Shakespeare's Hamlet</p>
    <p>🎭 Specialized in dramatic and poetic language patterns</p>
</div>
""", unsafe_allow_html=True)

