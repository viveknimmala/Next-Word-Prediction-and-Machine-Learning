# Project Name: Linguistic TensorFlow: Next-Word Prediction & English-Hindi Translation
## Overview
This project is divided into two main sections:

1. Next-Word Prediction: A deep learning model that predicts the next word in a sentence using the Brown Corpus dataset.
2. English-Hindi Translation: A machine translation model that translates English sentences to Hindi using a pre-trained Transformer model.
Additionally, a Django-based web application was developed to provide a user-friendly interface for utilizing these models.

## Table of Contents
Installation
Next-Word Prediction
English-Hindi Translation
Web Application
Usage

Next-Word Prediction
The Next-Word Prediction model is implemented in the NEXT_WORD_PREDICTION.ipynb file. It uses a Sequential model built with TensorFlow and trained on the Brown Corpus.

Key Steps:
1. Tokenization: The text data is tokenized using TensorFlow's Tokenizer.
2. Padding: The token sequences are padded to ensure uniform input size.
3. Model Architecture: The model includes an embedding layer followed by an LSTM layer and a dense layer with softmax activation.
4. Training: The model is trained using categorical cross-entropy as the loss function.

Machine_Translation.ipynb:

The English-Hindi Translation model is implemented in the Machine_Translation.ipynb file. This model leverages the pre-trained Helsinki-NLP/opus-mt-en-hi model from the Hugging Face Transformers library.

Key Steps:
1. Dataset: The IIT Bombay English-Hindi parallel corpus is used.
2. Preprocessing: The text is tokenized and prepared for translation.
3. Model Fine-tuning: The pre-trained model is fine-tuned using Seq2SeqTrainer.
4. Evaluation: The model's performance is evaluated using the BLEU metric.

## Web Application
A Django-based web application provides an interface for users to interact with the models.

### Files:
views.py: Handles the backend logic, including loading models and processing user input.
index.html: The frontend interface where users can input text and view the model's predictions or translations.

##  Usage
### Next-Word Prediction
Open the NEXT_WORD_PREDICTION.ipynb notebook.
Run the cells to load the model and predict the next word in a given sentence.

### Machine Translation
Open the Machine_Translation.ipynb notebook.
Run the cells to load the translation model and translate text from English to Hindi.

### Web Interface
Run the Django web server as described in the Web Application section.
Access the web interface to use either the Next-Word Prediction or English-Hindi Translation features.

### Model Saving and Loading
Both notebooks include steps to save the trained models and tokenizers for later use. Ensure these files are saved in the appropriate directories as shown in the notebooks.

### Results
Next-Word Prediction: The LSTM model effectively predicts the next word in a given sentence based on the patterns learned from the Brown corpus.
Machine Translation: The fine-tuned Helsinki-NLP model provides accurate translations from English to Hindi, evaluated using BLEU scores.

### Notes
Due to hardware limitations, the LSTM model training is limited to a subset of 20,000 sequences.
The machine translation model's training is also optimized for available computational resources by freezing specific layers