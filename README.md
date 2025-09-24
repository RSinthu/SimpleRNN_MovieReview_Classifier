# IMDB Movie Review Sentiment Analysis with Simple RNN

This project implements a sentiment analysis system for movie reviews using a Simple Recurrent Neural Network (RNN) trained on the IMDB dataset. The model can classify movie reviews as positive or negative, and includes a Streamlit web interface for easy interaction.

## 🎯 Project Overview

The project consists of two main components:
- **Model Training**: A Jupyter notebook that trains a Simple RNN on the IMDB dataset
- **Web Application**: A Streamlit app that provides an interactive interface for sentiment prediction

## 📁 Project Structure

```
RNN/
├── SimpleRNN.ipynb          # Model training notebook
├── main.py                  # Streamlit web application
├── simple_rnn_imdb.h5       # Trained model (generated after training)
└── README.md                # Project documentation
```

## 🏗️ Model Architecture

The Simple RNN model consists of:
- **Embedding Layer**: Maps vocabulary to 128-dimensional vectors
- **SimpleRNN Layer 1**: 128 units with return_sequences=True
- **SimpleRNN Layer 2**: 64 units (final output)
- **Dense Layer 1**: 32 units with ReLU activation
- **Output Layer**: 1 unit with sigmoid activation for binary classification

### Model Parameters:
- **Vocabulary Size**: 10,000 words
- **Embedding Dimensions**: 128
- **Max Sequence Length**: 500 tokens
- **Total Parameters**: 1,327,361 (5.06 MB)

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow streamlit numpy
```

### Training the Model

1. Open and run [`SimpleRNN.ipynb`](SimpleRNN.ipynb) to train the model:
   - Loads and preprocesses the IMDB dataset
   - Creates and trains the Simple RNN model
   - Saves the trained model as `simple_rnn_imdb.h5`

### Running the Web Application

1. Ensure the trained model (`simple_rnn_imdb.h5`) exists in the project directory
2. Run the Streamlit app:
   ```bash
   streamlit run main.py
   ```
3. Open your browser and navigate to the provided local URL
4. Enter a movie review and click "Classify" to get the sentiment prediction

## 📊 Model Performance

The model was trained with:
- **Optimizer**: Adam (learning_rate=0.001)
- **Loss Function**: Binary crossentropy
- **Metrics**: Accuracy
- **Early Stopping**: Patience of 5 epochs on validation loss
- **Training stopped at**: Epoch 11 due to early stopping

Final performance achieved validation accuracy of ~80% before early stopping.

## 💻 Usage Examples

### Using the Web Interface
1. Enter a movie review like: *"This movie was fantastic! Great acting and storyline."*
2. Click "Classify"
3. View the sentiment (Positive/Negative) and confidence score

### Using the Model Programmatically
```python
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('simple_rnn_imdb.h5')

# Preprocess and predict
text = "This movie was amazing!"
preprocessed = preprocess_text(text)
prediction = model.predict(preprocessed)
sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
```

## 🔧 Key Features

- **Text Preprocessing**: Handles word tokenization and sequence padding
- **Real-time Prediction**: Interactive web interface for instant sentiment analysis
- **Model Persistence**: Trained model saved for reuse
- **Early Stopping**: Prevents overfitting during training

## 📈 Model Training Details

The training process includes:
- **Data Loading**: IMDB dataset with 25,000 training and test samples
- **Preprocessing**: Sequence padding to uniform length of 500 tokens
- **Word Encoding**: Uses IMDB word index with offset (+3) for special tokens
- **Validation**: Uses test set for validation during training

## 📝 Files Description

- **[`SimpleRNN.ipynb`](SimpleRNN.ipynb)**: Complete model training pipeline with data loading, preprocessing, model creation, training, and testing
- **[`main.py`](main.py)**: Streamlit web application with text preprocessing function and user interface for sentiment prediction

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements or bug fixes.

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
