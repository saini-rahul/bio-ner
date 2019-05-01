from utilities import predict, load_saved_model
import nltk

nltk.download('punkt')

# load the model
model = load_saved_model()

while (1):
    try:
        input_sen = input("Type a sentence to predict slots on!\n")
    except:
        input_sen = None
    if input_sen:
        predict(input_sen, model)  
