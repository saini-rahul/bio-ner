from utilities import predict, load_saved_model
import nltk

nltk.download('punkt')

# load the model
model = load_saved_model()

while (1):
    input_sen = input("Type a sentence to predict slots on!\n")
    predict(input_sen, model)  
