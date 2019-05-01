from utilities import predict, load_saved_model
import nltk

nltk.download('punkt')

# load the model
model = load_saved_model()

while (1):
    try:
        input_sen = input("Type a sentence to predict slots on! Or enter exit to quit.\n")
    except:
        input_sen = None
    if input_sen.lower() == 'exit':
        break
    if input_sen:
        predict(input_sen, model)  
