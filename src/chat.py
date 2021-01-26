import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
import numpy as np
import json
import tensorflow as tf
import random
from tkinter import *
import time
import tkinter.messagebox


tokenizer=TreebankWordTokenizer()

with open("intent.json") as file:
    data=json.load(file)

words=[]
pat_match=[]
tag_match=[]
label=[]
train=[]
output=[]
for intents in data["intent"]:
    for pattern in intents["patterns"]:
        word=tokenizer.tokenize(pattern)
        words.extend(word)
        pat_match.append(pattern)
        tag_match.append(intents["tag"])
        
    if intents["tag"] not in label:
        label.append(intents["tag"]) 

words=[w.lower() for w in words if w != "?"]
stop_words=set(stopwords.words('English'))
words=[w for w in words if not w in stop_words]
words=sorted(list(set(words)))


empty=[]
empty=[0 for i in  range(len(label))]
    

wrd=[tokenizer.tokenize(i) for i in pat_match]

for index,p in enumerate(pat_match):
    presence=[]
    word=tokenizer.tokenize(p.lower())
    for w in words:
        if w in word:
            presence.append(1)
        else:
            presence.append(0)

    output_row=empty[:]
    output_row[label.index(tag_match[index])]=1
    train.append(presence)
    output.append(output_row)

train=np.array(train)
output=np.array(output)

model=tf.keras.models.load_model('E:\VIT\SEM5\CSE4022_NLP\project_\src')

def makeword(inp,words):
    bag=[0 for i in range(len(words))]
    inp_w=tokenizer.tokenize(inp) 
    inp_w=[w.lower() for w in inp_w if w != "?"]
    for w in inp_w:
        for index,p in enumerate(words):
            if p==w:
                bag[index]=1
    bag=np.array(bag)
    d=np.expand_dims(bag,0)
    return d

root=Tk()
root.title('VIT - Chatbot')
root.geometry('600x800')
canvas = Canvas(root, width = 500, height = 300)      
canvas.pack()      
img = PhotoImage(file="bot.png")      
canvas.create_image(5,5, anchor=NW, image=img)

class ChatInterface(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master

        self.tl_bg = "#EEEEEE"
        self.tl_bg2 = "#EEEEEE"
        self.tl_fg = "#000000"
        self.font = "Verdana 10"

        menu = Menu(self.master)
        self.master.config(menu=menu, bd=5)


    # File
        file = Menu(menu, tearoff=0)
        menu.add_cascade(label="File", menu=file)
        file.add_command(label="Clear Chat", command=self.clear_chat)
        file.add_command(label="Exit",command=self.chatexit)

    # Options
        options = Menu(menu, tearoff=0)
        menu.add_cascade(label="Options", menu=options)



        font = Menu(options, tearoff=0)
        options.add_cascade(label="Font", menu=font)
        font.add_command(label="Default",command=self.font_change_default)


        color_theme = Menu(options, tearoff=0)
        options.add_cascade(label="Color Theme", menu=color_theme)
        color_theme.add_command(label="Default",command=self.color_theme_default) 
        color_theme.add_command(label="Hacker",command=self.color_theme_hacker)

        self.text_frame = Frame(self.master, bd=6)
        self.text_frame.pack(expand=True, fill=BOTH)

        self.text_box_scrollbar = Scrollbar(self.text_frame, bd=0)
        self.text_box_scrollbar.pack(fill=Y, side=RIGHT)

        self.text_box = Text(self.text_frame, yscrollcommand=self.text_box_scrollbar.set, state=DISABLED,
                             bd=1, padx=6, pady=6, spacing3=8, wrap=WORD, bg=None, font="Verdana 10", relief=GROOVE,
                             width=10, height=1)
        self.text_box.pack(expand=True, fill=BOTH)
        self.text_box_scrollbar.config(command=self.text_box.yview)

        self.entry_frame = Frame(self.master, bd=1)
        self.entry_frame.pack(side=LEFT, fill=BOTH, expand=True)

        # entry field
        self.entry_field = Entry(self.entry_frame, bd=1, justify=LEFT)
        self.entry_field.pack(fill=X, padx=6, pady=6, ipady=3)


        self.send_button_frame = Frame(self.master, bd=0)
        self.send_button_frame.pack(fill=BOTH)

        self.send_button = Button(self.send_button_frame, text="Send", width=5, relief=GROOVE, bg='white',
                                  bd=1,command=lambda:self.chat(None),  activebackground="#FFFFFF",
                                  activeforeground="#000000")
        self.send_button.pack(side=LEFT, ipady=8)
        self.master.bind("<Return>",self.chat)

    def chat(self,mes):
        inp=self.entry_field.get()
        m="You: "+inp+"\n"
        self.text_box.configure(state=NORMAL)
        self.text_box.insert(END, m)
        self.text_box.configure(state=DISABLED)
        self.text_box.see(END)

        predict=model.predict(makeword(inp,words))  
        result=np.argmax(predict) 
        tag=label[result]

        for t in data['intent']:
            if t['tag']==tag:
                response=t['answer']
        botinp=random.choice(response)
        out="Bot: "+botinp+"\n"
        self.text_box.configure(state=NORMAL)
        self.text_box.insert(END,out)
        self.text_box.configure(state=DISABLED)
        self.text_box.see(END)
        self.entry_field.delete(0,END)



    def clear_chat(self):
        self.text_box.config(state=NORMAL)
        self.last_sent_label(date="No messages sent.")
        self.text_box.delete(1.0, END)
        self.text_box.delete(1.0, END)
        self.text_box.config(state=DISABLED)

    def chatexit(self):
        exit()

    def font_change_default(self):
        self.text_box.config(font="Verdana 10")
        self.entry_field.config(font="Verdana 10")
        self.font = "Verdana 10"

    

    def color_theme_default(self):
        self.master.config(bg="#EEEEEE")
        self.text_frame.config(bg="#EEEEEE")
        self.entry_frame.config(bg="#EEEEEE")
        self.text_box.config(bg="#FFFFFF", fg="#000000")
        self.entry_field.config(bg="#FFFFFF", fg="#000000", insertbackground="#000000")
        self.send_button_frame.config(bg="#EEEEEE")
        self.send_button.config(bg="#FFFFFF", fg="#000000", activebackground="#FFFFFF", activeforeground="#000000")

        self.tl_bg = "#FFFFFF"
        self.tl_bg2 = "#EEEEEE"
        self.tl_fg = "#000000"

    
    # Hacker
    def color_theme_hacker(self):
        self.master.config(bg="#0F0F0F")
        self.text_frame.config(bg="#0F0F0F")
        self.entry_frame.config(bg="#0F0F0F")
        self.text_box.config(bg="#0F0F0F", fg="#33FF33")
        self.entry_field.config(bg="#0F0F0F", fg="#33FF33", insertbackground="#33FF33")
        self.send_button_frame.config(bg="#0F0F0F")
        self.send_button.config(bg="#0F0F0F", fg="#FFFFFF", activebackground="#0F0F0F", activeforeground="#FFFFFF")

        self.tl_bg = "#0F0F0F"
        self.tl_bg2 = "#0F0F0F"
        self.tl_fg = "#33FF33"

    

    # Default font and color theme
    def default_format(self):
        self.font_change_default()
        self.color_theme_default()    




a = ChatInterface(root)
root.mainloop()