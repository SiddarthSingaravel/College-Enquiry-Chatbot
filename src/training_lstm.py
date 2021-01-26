import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Flatten, Conv1D, MaxPooling1D,SpatialDropout1D,LSTM,Dropout

tokenizer=TreebankWordTokenizer()

with open("intent.json") as file:
    data=json.load(file)



def training():
    
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
    stop_words=set(stopwords.words('English'))
    words=[w.lower() for w in words if w != "?"]
    words=[w for w in words if not w in stop_words]
    words=sorted(list(set(words)))


    empty=[]
    empty=[0 for i in range(len(label))]  
    

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


    print(words)
    print(presence)
    #print(output)
    #print(tag_match)
    #print(label)
    #print(empty)
            
    print(train.shape) #293,339
    
    model = Sequential()
    model.add(Embedding(train.shape[1],300,input_length=train.shape[1]))
    model.add(tf.keras.layers.GRU(10, return_sequences=True))
    model.add(tf.keras.layers.SimpleRNN(10))
    model.add(Dropout(0.3))
    model.add(Dense(5, activation = 'softmax'))
    model.add(tf.keras.layers.Dense(len(label),activation=tf.nn.softmax))

    
    

    model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
    
    model.fit(train,output,epochs = 200)

    model_lstm.save('E:\VIT\SEM5\CSE4022_NLP\project_\src')

training()
