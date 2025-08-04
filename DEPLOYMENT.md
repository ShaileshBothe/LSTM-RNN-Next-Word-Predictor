# 🚀 Streamlit Cloud Deployment Guide

## Prerequisites

1. **GitHub Account**: Make sure your code is pushed to a GitHub repository
2. **Streamlit Cloud Account**: Sign up at [share.streamlit.io](https://share.streamlit.io)

## Step-by-Step Deployment

### 1. Prepare Your Repository

Make sure your repository contains these files:
```
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies (use requirements-simple.txt if needed)
├── .streamlit/config.toml # Streamlit configuration
├── hamlet.txt            # Training data
├── next_word_lstm.h5     # Trained model (optional)
├── tokenizer.pickle      # Saved tokenizer (optional)
└── README.md            # Project documentation
```

### 2. Fix Pip Installation Issues

If you encounter pip installation errors:

#### Option A: Use Minimal Requirements (Recommended)
Use the minimal requirements file:
```bash
cp requirements-minimal.txt requirements.txt
```

#### Option B: Use Simple Requirements
Use the simple requirements file:
```bash
cp requirements-simple.txt requirements.txt
```

#### Option C: Manual Fix
If you still get errors, manually edit `requirements.txt` to contain only:
```
streamlit
tensorflow
```

### 3. Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**: Visit [share.streamlit.io](https://share.streamlit.io)
2. **Sign in**: Use your GitHub account
3. **New App**: Click "New app"
4. **Repository**: Select your GitHub repository
5. **Main file path**: Enter `app.py`
6. **Python version**: Select Python 3.9
7. **Deploy**: Click "Deploy!"

### 4. Handle Model Loading Issues

If you encounter model loading errors:

#### Option A: Use Model Recreation (Recommended)
The app will automatically recreate the model if loading fails. This is the safest option.

#### Option B: Retrain and Save Model
If you want to use the saved model:

1. **Retrain locally** with the same TensorFlow version as Streamlit Cloud
2. **Save the model** using:
   ```python
   model.save('next_word_lstm.h5', save_format='h5')
   ```
3. **Upload the new model** to your repository

### 5. Troubleshooting Common Issues

#### Issue: Pip Installation Error
**Solution**: 
1. Use `requirements-simple.txt` instead of `requirements.txt`
2. Or try the specific versions in the current `requirements.txt`
3. Make sure Python version is 3.9

#### Issue: Model Loading Error
**Solution**: The app will automatically recreate the model. This is normal and expected.

#### Issue: Tokenizer Loading Error
**Solution**: The app will recreate the tokenizer from `hamlet.txt`. This is also normal.

#### Issue: Memory Issues
**Solution**: 
- Reduce model size in the recreation code
- Use smaller embedding dimensions
- Reduce LSTM units

#### Issue: Build Failures
**Solution**:
- Check that all files are in the repository
- Verify `requirements.txt` syntax
- Ensure `app.py` is the main file
- Try Python 3.9 instead of 3.10

### 6. Alternative Deployment Steps

If you continue to have issues:

1. **Use minimal requirements**:
   ```txt
   streamlit
   tensorflow
   numpy
   ```

2. **Remove optional dependencies** from requirements.txt

3. **Test locally first**:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

### 7. Update Your README

Once deployed, update your README.md with the actual deployment URL:

```markdown
## 🌟 Live Demo

**[🚀 Deploy on Streamlit Cloud](https://your-actual-app-name.streamlit.app)**
```

### 8. Monitor Your App

- **Check logs**: Click "Manage app" in the bottom right
- **View errors**: Check the logs for detailed error messages
- **Test functionality**: Try the working examples

## Expected Behavior

After deployment, you should see:
1. ✅ Model loaded successfully! (or Model recreated successfully!)
2. ✅ Tokenizer loaded from pickle! (or Tokenizer recreated successfully!)
3. The main interface with working examples

## Performance Notes

- **First load**: May take 30-60 seconds to recreate model/tokenizer
- **Subsequent loads**: Much faster due to caching
- **Memory usage**: ~500MB-1GB depending on model size

## Support

If you continue to have issues:
1. Check the Streamlit Cloud logs
2. Verify all files are in the repository
3. Test locally first with `streamlit run app.py`
4. Consider using the model recreation approach (most reliable)
5. Try the simple requirements file 