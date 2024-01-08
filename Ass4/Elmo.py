import nltk
nltk.download('stopwords')
nltk.download('punkt')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import re
import torchtext
from torchtext.data import get_tokenizer
import json
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
import json

# loading the data using datasets
from datasets import load_dataset
dataset = load_dataset("sst", "default")

# extracting test,train,validation data
test=dataset['test']
train=dataset['train']
validation=dataset['validation']

# wtiting sentence and labels to another files
with open('/content/drive/MyDrive/Ass4NLP/test.json', 'w') as f:
    # write the dictionary to the file as JSON data
    for x in test:
      line={}
      line['sentence']=x['sentence']
      line['label']=x['label']
      json.dump(line, f)
      f.write('\n')

# reading from the files and storing in lists
train_x=[]
train_y=[]

with open('/content/drive/MyDrive/Ass4NLP/train.json', 'r') as file:
    # write the dictionary to the file as JSON data
        for line in file:
          data = json.loads(line)
          train_x.append(data['sentence'])
          train_y.append(data['label'])
          
val_x=[]
val_y=[]

with open('/content/drive/MyDrive/Ass4NLP/validation.json', 'r') as file:
    # write the dictionary to the file as JSON data
        for line in file:
          data = json.loads(line)
          val_x.append(data['sentence'])
          val_y.append(data['label'])

test_x=[]
test_y=[]

with open('/content/drive/MyDrive/Ass4NLP/test.json', 'r') as file:
    # write the dictionary to the file as JSON data
        for line in file:
          data = json.loads(line)
          test_x.append(data['sentence'])
          test_y.append(data['label'])


stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words("english"))
def preprocess(text):
    filtered_tokens = []
    for token in text:
        if token.lower() not in stop_words:
            filtered_tokens.append((stemmer.stem(token)))
           
    return filtered_tokens

tokenizer = get_tokenizer('basic_english')
tokenized_train=[preprocess(tokenizer(sent)) for sent in train_x]
tokenized_val=[preprocess(tokenizer(sent)) for sent in val_x]
tokenized_test=[preprocess(tokenizer(sent)) for sent in test_x]

# for creating vocabulay and finding max length
def create_vocabulary(data):
  unigrams={}
  max_length=0
  for sent in data:
    if max_length< len(sent):
      max_length=len(sent)
    for word in sent:
      if word in unigrams:
        unigrams[word]+=1
      else :
        unigrams[word]=1
  vocab={}
  vocab['<unk>']=0
  vocab['<pad>']=1
  for word in unigrams:
    if unigrams[word] > 2:
      vocab[word]=len(vocab)
  return vocab,max_length

vocabulary,max_length=create_vocabulary(tokenized_train)
# print(len(vocabulary))


# converting words to index
def word_to_index(data):
  index=[]
  for sent in data:
    dummy=[]
    for word in sent :
      if word in vocabulary:
        dummy.append(vocabulary[word])
      else :
        dummy.append(vocabulary['<unk>'])
    index.append(dummy)
  return index

# print(tokenized_train[99])
train_data=word_to_index(tokenized_train)
val_data=word_to_index(tokenized_val)
test_data=word_to_index(tokenized_test)
# print(val_data[0])

def padding(data,max_length):
  padded_sent=[]
  padded_sent=[sent + [vocabulary['<pad>']]*(max_length-len(sent)) for sent in data]
  return padded_sent


class Datasets():
  def __init__(self, data, labels, max_len):
    self.data = padding(data,max_len)
    self.labels=labels
    self.batch_size=64
    self.inputs=self.data
    self.context =[sent[1:] for sent in self.data]
    self.target=[sent[:-1] for sent in self.data]
    self.batchs=self.making_batchs()

  def making_batchs(self):
     self.target=torch.tensor(self.target)
     self.labels=torch.tensor(self.labels)
     self.context=torch.tensor(self.context)
     self.inputs=torch.tensor(self.inputs)
     batch_target=torch.split(self.target,self.batch_size)
     batch_input=torch.split(self.inputs,self.batch_size)
     batch_context=torch.split(self.context,self.batch_size)
     batch_labels=torch.split(self.labels,self.batch_size)
     batchs=[]
     for batch in batch_labels:
       batchs.append((batch_context,batch_target,batch_input,batch_labels))
     return batchs

train_loader=Datasets(train_data,train_y,150)
validation_loader=Datasets(val_data,val_y,150)
test_loader=Datasets(test_data,test_y,150)

from torchtext.vocab import GloVe
glove = GloVe(name='6B', dim=100)

class Sentiment(nn.Module):
    def __init__(self, vocabulary, embedding_dim, hidden_dim, num_classes):
        super(Sentiment, self).__init__()
        self.vocab_size=len(vocabulary)
        self.e_dim=embedding_dim
        self.h_dim=hidden_dim
        self.vocabulary=vocabulary

        self.lstm_forward1 = nn.LSTM(self.e_dim,self.h_dim, batch_first=True)
        self.lstm_forward2 = nn.LSTM(self.h_dim, self.h_dim, batch_first=True)

        self.lstm_backward1 = nn.LSTM(self.e_dim,self.h_dim, batch_first=True)
        self.lstm_backward2 = nn.LSTM(self.h_dim, self.h_dim, batch_first=True)

        self.Linear = nn.Linear(self.h_dim, self.vocab_size)
        self.Embedding_layer = self.Em_layer()
        self.Embedding_layer.weight = nn.Parameter(self.Embedding_layer.weight, requires_grad=True)
        self.weights = nn.Parameter(torch.tensor([0.33, 0.33, 0.33]), requires_grad=True)
        self.clasifier=nn.Linear(self.h_dim*2,num_classes)
        # self.dropout = nn.Dropout(0.5)

    def Em_layer(self):
      embedding_matrix = torch.zeros(self.vocab_size,self.e_dim)
      for word, index in self.vocabulary.items():
        if word in glove.stoi:
            embedding_matrix[index] = glove.vectors[glove.stoi[word]]
      return nn.Embedding.from_pretrained(embedding_matrix, freeze=False, padding_idx=self.vocabulary['<pad>'])

    def forward(self,input):
      embeddings = self.Embedding_layer(input)
      h1_for,cell1 = self.lstm_forward1(embeddings)
      h2_for,cell2 = self.lstm_forward2(embeddings)

      rev_embeddings = torch.flip(embeddings,[1])

      h1_back, (_, _) = self.lstm_backward1(rev_embeddings)
      h2_back, (_, _) = self.lstm_backward2(rev_embeddings)
      

      h1_back_rev = torch.flip(h1_back, [1])
      h2_back_rev = torch.flip(h2_back, [1])
      return (h2_for, h1_for, h2_back_rev, h1_back_rev)
    def forward_output(self, batch):
        h2_for, h1_for, h2_back, h1_back = self.forward(batch)
        output_for = self.Linear(h2_for)
        output_back = self.Linear(h2_back)
        return (output_for, output_back)
    def classification(self,input):
      h2_for, h1_for, h2_back, h1_back = self.forward(input)
      embeddings=self.Embedding_layer(input)
      layer2_h= torch.cat((h2_for, h2_back), 2)
      layer1_h = torch.cat((h1_for, h1_back), 2)
      same_dim_em = embeddings.repeat((1, 1, 2))
      final_em=layer1_h*self.weights[0]+layer2_h*self.weights[1]+same_dim_em*self.weights[2]
      output=self.clasifier(final_em)
      return output


sentiment_A=Sentiment(vocabulary,100,100,2)
optimizer = optim.Adam(sentiment_A.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
max_length=150

def train_elmo(model,epochs,train_loader,validation_loader):

  for epoch in range(epochs):
    train_loss=0
    sentiment_A.train()
    
    for batch,(context,target,data,labels) in enumerate(train_loader.batchs) :
      optimizer.zero_grad()
      # print(np.shape(context))
      (f_output,b_output) = model.forward_output(data[batch])
      forward_output = f_output[:, :max_length-1, :]
      backward_output = b_output[:, :max_length-1, :]
      # print(forward_output)
      forward_output=forward_output.reshape(-1,forward_output.shape[2])
      backward_output=backward_output.reshape(-1,backward_output.shape[2])
      
      for_target = data[batch][:,1:]
      for_target=for_target.reshape(-1)
      back_target = data[batch][:,:-1]
      back_target=back_target.reshape(-1)
      loss=criterion(forward_output,for_target)+criterion(backward_output,back_target)
      train_loss=loss.item()+train_loss
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      # print(loss.item(),"loss")
    print(train_loss/len(train_loader.batchs),"train")
    val_loss=0
    for batch,(context,target,data,labels) in enumerate(validation_loader.batchs) :
      # print(np.shape(context))
      (f_output,b_output) = model.forward_output(data[batch])
      forward_output = f_output[:, :max_length-1, :]
      backward_output = b_output[:, :max_length-1, :]
      # print(forward_output)
      forward_output=forward_output.reshape(-1,forward_output.shape[2])
      backward_output=backward_output.reshape(-1,backward_output.shape[2])
      
      for_target = data[batch][:,1:]
      for_target=for_target.reshape(-1)
      back_target = data[batch][:,:-1]
      back_target=back_target.reshape(-1)
      loss=criterion(forward_output,for_target)+criterion(backward_output,back_target)
      val_loss=loss.item()+val_loss
      # print(loss)
      
      # print(loss.item(),"loss")
    print(val_loss/len(validation_loader.batchs),"validation")

train_elmo(sentiment_A,5,train_loader,validation_loader)


def classification_task(model,epochs,train_loader):
  for epoch in range(epochs):
    train_loss=0
    model.train()
    for batch,(context,target,data,labels) in enumerate(train_loader.batchs) :
      optimizer.zero_grad()
      output = model.clasifier(data[batch])
      target=torch.round(labels[batch])
      loss=criterion(output,target)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      train_loss+=loss.item
    print(train_loss/len(train_loader.batchs),"train")
    start=100
    for epoch in range(epochs):
      val_loss=0
      model.eval()
      for batch,(context,target,data,labels) in enumerate(validation_loader.batchs) :
        optimizer.zero_grad()
        output = model.clasifier(data[batch])
        target=torch.round(labels[batch])
        loss=criterion(output,target)
        val_loss+=loss.item
      if start > val_loss:
        start=val_loss
        torch.save(model.state_dict(), 'elmo.pt')
        print('Model updated and saved')
      print(train_loss/len(train_loader.batchs),"train")
classification_task(sentiment_A,5,train_loader)


from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

def test(model,test_loader):
    actual=[]
    predict=[]
    for batch,(context,target,data,labels) in enumerate(test_loader.batchs) :
        output = model.clasifier(data[batch])
        output=torch.nn.functional.softmax(output,dim=0).cpu().detach().numpy()
        answer=[np.argmax(output[i]) for i in range(len(output))]
        tags=labels[batch].view(-1)
        actual=actual+tags.tolist()
        predict=predict + answer
    print(classification_report(predict,actual,zero_division=0))
    fpr, tpr, thresholds = roc_curve(actual,predict)
    auc = roc_auc_score(actual,predict)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    cm = confusion_matrix(actual,predict)
    print(cm)

test(sentiment_A,test_loader)

# torch.save(sentiment_A.state_dict(),'/home/dell/NLP/ass4/model.pth')
sentiment_A=Sentiment(vocabulary,100,100,2)
sentiment_A.load_state_dict(torch.load('/home/dell/NLP/Assignment2/ass4/model.pth'))
   
