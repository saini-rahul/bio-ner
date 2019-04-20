import tkinter

from tkinter import *
from utilities import predict, load_saved_model
import nltk

def predict_slots():
  varContent = t1.get("1.0", "end-1c")
  print ("Predicting...")
  original_tokens, return_tags = predict(varContent, model)
  zipped_tags =  list(zip( original_tokens, *return_tags)) # a list of tokens, where token [0] is original text
  print (zipped_tags)
  listbox_word.delete(0, END)
  listbox_token.delete(0, END)

  for r, tokens in enumerate(zipped_tags):
    if(len(tokens) >= 2):
        word = tokens[0]
        tag = " / ".join(tokens[1:])
        listbox_word.insert(END, word)
        listbox_token.insert(END, tag)
    else:
        word = tokens[0]
        listbox_word.insert(END, word)
        listbox_token.insert(END, "No relevant slots!")

        
  print ('Printed on GUI.')
nltk.download('punkt')

# load the model
model = load_saved_model()


root = Tk()
root.resizable(width=FALSE, height=FALSE)
root.geometry("600x300")
root.config(bg="grey")


#create containers
top_frame = Frame(root, bg="cyan", width =600, height = 100)
top_frame.grid(row=0)
top_frame.pack_propagate(0)
bottom_frame = Frame(root, bg="green", width=600, height = 200)
bottom_frame.grid(row=1)

submit_button = Button(top_frame, text ='Predict Slots', command = predict_slots)
l1=Label(top_frame, text='Please type a sentence (about flights, movies, restaurants!) below, and press <Predict Slots>.')
t1 = Text(top_frame,bd=0,font='Fixdsys -14')

submit_button.pack()
l1.pack()
t1.pack()

g1 = Label(bottom_frame, text="Word")
g2 = Label(bottom_frame, text="Slots")
g1.grid ( row=0, column=0)
g2.grid (row=0, column=1)
listbox_word = Listbox(bottom_frame)
listbox_word.grid (row=1, column=0)
listbox_token = Listbox(bottom_frame)
listbox_token.grid (row=1, column=1)

root.mainloop()