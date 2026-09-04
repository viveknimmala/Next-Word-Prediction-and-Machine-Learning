from django.shortcuts import render
from django.http import JsonResponse
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.models import load_model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load the word prediction model
model_path = 'C:/Users/RohithSai/OneDrive - Indian Institute of Technology Guwahati/documents/PROJECTS/MAIN PROJECTS/MACHINE TRANSLATION/my_model.h5'
model = load_model(model_path)

# Load the word prediction tokenizer
with open(r'C:\Users\RohithSai\OneDrive - Indian Institute of Technology Guwahati\documents\PROJECTS\MAIN PROJECTS\MACHINE TRANSLATION\tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the translation model and tokenizer
translation_model_path = r'C:\Users\RohithSai\OneDrive - Indian Institute of Technology Guwahati\documents\PROJECTS\MAIN PROJECTS\MACHINE TRANSLATION\finetuned-nlp-en-hi'
translation_tokenizer = AutoTokenizer.from_pretrained(translation_model_path)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(translation_model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
translation_model.to(device)

def index(request):
    return render(request, 'predictor/index.html')

def predict_word(request):
    if request.method == 'POST':
        input_text = request.POST.get('input_text', '')

        if not input_text:
            return JsonResponse({'error': 'No input text provided.'}, status=400)

        try:
            # Tokenize and pad the input
            tokens = tokenizer.texts_to_sequences([input_text])[0]
            tokens_padded = pad_sequences([tokens], maxlen=167-1, padding='pre')

            # Make a prediction
            predictions = model.predict(tokens_padded)

            # Get the predicted index and map it to the corresponding word
            predicted_index = np.argmax(predictions, axis=-1)[0]
            next_word = decode_predictions(predicted_index)

            return JsonResponse({'prediction': next_word})

        except Exception as e:
            print(f"An error occurred: {e}")
            return JsonResponse({'error': 'Error occurred. Please try again.'}, status=500)

    return JsonResponse({'error': 'Invalid request method. Only POST is allowed.'}, status=405)

def decode_predictions(predicted_index):
    try:
        # Map the predicted index to the word
        word_index = tokenizer.word_index
        reverse_word_index = {v: k for k, v in word_index.items()}
        next_word = reverse_word_index.get(predicted_index, 'Unknown')
        return next_word

    except Exception as e:
        print(f"Error decoding predictions: {e}")
        return 'Decoding error'

def translate_text(request):
    if request.method == 'POST':
        input_text = request.POST.get('input_text', '')

        if not input_text:
            return JsonResponse({'error': 'No input text provided.'}, status=400)

        try:
            inputs = translation_tokenizer(input_text, return_tensors="pt").to(device)
            translated_tokens = translation_model.generate(**inputs, max_length=256)
            translated_text = translation_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

            return JsonResponse({'translation': translated_text})

        except Exception as e:
            print(f"An error occurred during translation: {e}")
            return JsonResponse({'error': 'Translation error occurred. Please try again.'}, status=500)

    return JsonResponse({'error': 'Invalid request method. Only POST is allowed.'}, status=405)



# from django.shortcuts import render
# from django.http import JsonResponse
# import tensorflow as tf
# import pickle
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import numpy as np
# from tensorflow.keras.models import load_model

# def index(request):
#     return render(request, 'predictor/index.html')

# # Load the Keras model
# model_path = 'C:/Users/RohithSai/OneDrive - Indian Institute of Technology Guwahati/documents/PROJECTS/MAIN PROJECTS/OVERALL MACHINE TRANSLATION/my_model.h5'
# model = load_model(model_path)

# # Load the tokenizer
# with open(r'C:\Users\RohithSai\OneDrive - Indian Institute of Technology Guwahati\documents\PROJECTS\MAIN PROJECTS\OVERALL MACHINE TRANSLATION\tokenizer.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)

# def predict_word(request):
#     if request.method == 'POST':
#         input_text = request.POST.get('input_text', '')
        
#         if not input_text:
#             return JsonResponse({'error': 'No input text provided.'}, status=400)

#         try:
#             # Tokenize and pad the input
#             tokens = tokenizer.texts_to_sequences([input_text])[0]
#             tokens_padded = pad_sequences([tokens], maxlen=167-1, padding='pre')

#             # Make a prediction
#             predictions = model.predict(tokens_padded)

#             # Get the predicted index and map it to the corresponding word
#             predicted_index = np.argmax(predictions, axis=-1)[0]
#             next_word = decode_predictions(predicted_index)

#             return JsonResponse({'prediction': next_word})

#         except Exception as e:
#             print(f"An error occurred: {e}")
#             return JsonResponse({'error': 'Error occurred. Please try again.'}, status=500)

#     return JsonResponse({'error': 'Invalid request method. Only POST is allowed.'}, status=405)


# def decode_predictions(predicted_index):
#     try:
#         # Map the predicted index to the word
#         word_index = tokenizer.word_index
#         reverse_word_index = {v: k for k, v in word_index.items()}
#         next_word = reverse_word_index.get(predicted_index, 'Unknown')
#         return next_word

#     except Exception as e:
#         print(f"Error decoding predictions: {e}")
#         return 'Decoding error'

