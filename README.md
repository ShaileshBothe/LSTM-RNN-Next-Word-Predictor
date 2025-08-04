# 🧠 LSTM RNN Next Word Predictor

A sophisticated word prediction application powered by Long Short-Term Memory (LSTM) neural networks, trained on Shakespeare's Hamlet to predict the next word in a sequence.

## 🌟 Live Demo

**[🚀 Deploy on Streamlit Cloud](https://your-app-name-here.streamlit.app)**

*Replace the link above with your actual Streamlit Cloud deployment URL*

## 📋 Project Overview

This project demonstrates the power of LSTM neural networks in natural language processing by creating a word prediction system. The model is trained on Shakespeare's Hamlet and can predict the next word in a given sequence with remarkable accuracy.

## ✨ Features

- **🤖 AI-Powered Predictions**: LSTM neural network for accurate word prediction
- **📚 Shakespeare Training**: Model trained on Hamlet for literary language patterns
- **🎯 Interactive Examples**: Click-to-test working examples
- **📊 Performance Metrics**: Real-time accuracy and confidence indicators
- **🎨 Modern UI**: Beautiful Streamlit interface with gradient designs
- **📱 Responsive Design**: Works on desktop and mobile devices

## 🏗️ Model Architecture

```
Embedding Layer (100-dim) → LSTM Layer 1 (150 units) → Dropout (20%) → LSTM Layer 2 (100 units) → Dense Layer (Softmax)
```

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- TensorFlow 2.x
- Streamlit

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd LSTM-RNN-Next-Word-Prediction-Project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8501`

## 📁 Project Structure

```
LSTM RNN Nest Word Prediction Project/
├── app.py                 # Main Streamlit application
├── experiement.ipynb      # Jupyter notebook with model training
├── hamlet.txt            # Training data (Shakespeare's Hamlet)
├── next_word_lstm.h5     # Trained LSTM model
├── tokenizer.pickle      # Saved tokenizer (optional)
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🎯 How to Use

1. **Enter Text**: Type a sequence of words in the input field
2. **Get Prediction**: Click "Predict Next Word" to see the AI's prediction
3. **Try Examples**: Click any working example to see instant predictions
4. **Custom Testing**: Use the form to test your own phrases

## 💡 Working Examples

- **"To be or not to"** → Shakespeare's famous line
- **"The quick brown"** → Common phrase continuation
- **"In the beginning"** → Biblical style text
- **"Once upon a"** → Fairy tale opening
- **"Hamlet shall never"** → Shakespeare's Hamlet reference

## 🛠️ Technologies Used

- **TensorFlow/Keras**: Deep learning framework
- **LSTM**: Long Short-Term Memory neural networks
- **Streamlit**: Web application framework
- **NLTK**: Natural language processing
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation

## 📊 Model Performance

- **Training Data**: Shakespeare's Hamlet (~3,000+ unique words)
- **Architecture**: Dual LSTM layers with embedding
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Training Epochs**: 5

## 🚀 Deployment

### Streamlit Cloud Deployment

1. **Push to GitHub**: Upload your code to a GitHub repository
2. **Connect to Streamlit Cloud**: 
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository
3. **Deploy**: Click deploy and wait for the build to complete
4. **Update Link**: Replace the deployment link in this README

### Local Deployment

```bash
streamlit run app.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Shakespeare's Hamlet**: Training data source
- **TensorFlow/Keras**: Deep learning framework
- **Streamlit**: Web application platform
- **NLTK**: Natural language processing library


---

⭐ **Star this repository if you found it helpful!** 