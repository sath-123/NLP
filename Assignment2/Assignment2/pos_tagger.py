import torch
from torch import nn, tensor
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm
import torch.optim as optim
from torch import nn
import math
from sklearn.metrics import classification_report
import numpy as np
import re
from sklearn.metrics import classification_report
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class TagsAndWords(nn.Module):
  def __init__(self,filepath):
    super(TagsAndWords,self).__init__()
    self.hidden_size=200
    self.emb_dim=300
    
    self.filename=filepath
    self.sentences=[]
    self.Postags=[]
    self.batches=32
    
    words=[]
    Tags=[]
    index=-1
    maxlength=-1
    with open(self.filename, 'r', encoding='utf-8') as file:
      sent=[]
      tags=[]
      for line in file:
        if not line.startswith('#'):
          line = line.strip()
          if line:
            fields = line.split('\t')
            ind=fields[0]
            if(index>int(ind)):
              self.sentences.append(sent)
              self.Postags.append(tags)
              sent=[]
              tags=[]
              if maxlength<index:
                maxlength=index
            else :
              sent.append(fields[1])
              tags.append(fields[3])
              words.append([fields[1]])
              Tags.append([fields[3]])
            index=int(ind)

      self.sentences.append(sent)
      self.Postags.append(tags)     
      self.maxlength=maxlength
      self.vocabulary,self.POStags=self.get_unique(words,Tags)
      self.equalsize=self.Makeequalsize()
      self.Esizesent,self.Esizetag=self.Makeequalsize()
      self.batches=self.Makebatch()
      self.embedding = nn.Embedding(len(self.vocabulary),self.emb_dim)
      self.lstm = nn.LSTM(input_size=self.emb_dim, hidden_size=self.hidden_size)
      self.answer = nn.Linear(self.hidden_size,len(self.POStags))
    
      
  def get_unique(self,words,Tags):
        # print(words)
        vocab = build_vocab_from_iterator(words, min_freq=1,
                                           specials=['<PAD>', '<UNK>'],
                                           special_first=False)
        vocab.set_default_index(vocab.get_stoi()['<UNK>'])

        postags = build_vocab_from_iterator(Tags, min_freq=1,
                                           specials=['<UNKTAG>','<PAD>'],
                                           special_first=False)
        postags.set_default_index(postags.get_stoi()['<UNKTAG>'])
      
        return vocab,postags


  def Makeequalsize(self):
    length=len(self.sentences)
    sentwords=[]
    senttags=[]
    for i in range(0,length):
      length1=len(self.sentences[i])
      word_index=[]
      tag_index=[]
      for j in range(0,length1):
          word_index.append(self.vocabulary[self.sentences[i][j]])
          tag_index.append(self.POStags[self.Postags[i][j]])
      words= word_index + [self.vocabulary['<PAD>']]*(self.maxlength-length1)
      tags= tag_index + [self.POStags['<PAD>']]*(self.maxlength-length1)
      sentwords.append(words)
      senttags.append(tags)
      
    # print(answer)

    return sentwords,senttags

  def Makebatch(self):
    # print(self.Esize)
    self.Esizesent=torch.tensor(self.Esizesent)
    self.Esizetag=torch.tensor(self.Esizetag)
    # print(self.Esize)
    batch_wiseSen=torch.split(self.Esizesent,32)
    batch_wiseTag=torch.split(self.Esizetag,32)

    batchs=[]
    for i in range(0,len(batch_wiseSen)):
      batchs.append((batch_wiseSen[i],batch_wiseTag[i]))
    # print(len(batchs))
    return batchs
  
  def forward(self,context):
    embed_output=self.embedding(context)
    out,state=self.lstm(embed_output)
    final=self.answer(out)
    # print(final)
    return final

  

# class POSMODEL(nn.Module):
#   def __init__(self, vocab_size,pos_size):
#     super(POSMODEL,self).__init__()
#     self.hidden_size=100
#     self.emb_dim=200
#     self.embedding = nn.Embedding(vocab_size,self.emb_dim)
#     self.lstm = nn.LSTM(input_size=self.emb_dim, hidden_size=self.hidden_size)
#     self.answer = nn.Linear(self.hidden_size,pos_size)
    
#   def forward(self,context):
#     embed_output=self.embedding(context)
#     out,state=self.lstm(embed_output)
#     final=self.answer(out)
#     # print(final)
#     return final


class ValTags():
  def __init__(self,filepath,traindata):
    self.filename=filepath
    self.sentences=[]
    self.Postags=[]
    words=[]
    Tags=[]
    index=-1
    maxlength=-1
    with open(self.filename, 'r', encoding='utf-8') as file:
      sent=[]
      tags=[]
      for line in file:
        if not line.startswith('#'):
          line = line.strip()
          if line:
            fields = line.split('\t')
            ind=fields[0]
            if(index>int(ind)):
              self.sentences.append(sent)
              self.Postags.append(tags)
              sent=[]
              tags=[]
              if maxlength<index:
                maxlength=index
            else :
              sent.append(fields[1])
              tags.append(fields[3])
              words.append([fields[1]])
              Tags.append([fields[3]])
            index=int(ind)

      self.sentences.append(sent)
      self.Postags.append(tags)     
      self.maxlength=maxlength
      self.equalsize=self.Makeequalsize()
      self.Esizesent,self.Esizetag=self.Makeequalsize()
      self.batches=self.Makebatch()

  def Makeequalsize(self):
    length=len(self.sentences)
    sentwords=[]
    senttags=[]
    for i in range(0,length):
      length1=len(self.sentences[i])
      word_index=[]
      tag_index=[]
      for j in range(0,length1):
          word_index.append(traindata.vocabulary[self.sentences[i][j]])
          tag_index.append(traindata.POStags[self.Postags[i][j]])
      words= word_index + [traindata.vocabulary['<PAD>']]*(self.maxlength-length1)
      tags= tag_index + [traindata.POStags['<PAD>']]*(self.maxlength-length1)
      sentwords.append(words)
      senttags.append(tags)
      
    # print(answer)

    return sentwords,senttags

  def Makebatch(self):
    # print(self.Esize)
    self.Esizesent=torch.tensor(self.Esizesent)
    self.Esizetag=torch.tensor(self.Esizetag)
    # print(self.Esize)
    batch_wiseSen=torch.split(self.Esizesent,32)
    batch_wiseTag=torch.split(self.Esizetag,32)

    batchs=[]
    for i in range(0,len(batch_wiseSen)):
      batchs.append((batch_wiseSen[i],batch_wiseTag[i]))
    # print(len(batchs))
    return batchs
    

# print(traindata.Esize)

def training(val,model,L_rate,epochs):
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(),lr=L_rate)
  initial_loss=-math.inf
  for i in range(epochs):
    actual=[]
    predict=[]
    model.train()
    loss_for_epoch=0
    # print("entered")
    batchs=0
    for batch,(words,prediction) in enumerate(model.batches):
      optimizer.zero_grad() # set the gradients to zero before starting to do backpropragation 
      predicted_one=model(words)
      predicted_one=predicted_one.view(-1,predicted_one.shape[2])
      prediction=prediction.view(-1)
      loss=criterion(predicted_one,prediction)
      loss.backward()
      optimizer.step()
      loss_for_epoch+=loss.item()
      batchs+=1
    loss_for_epoch=loss_for_epoch/batchs
    initial_loss=loss_for_epoch
    # print(loss_for_epoch)


    model.eval()
    val_loss=0
    actual=[]
    predict=[]
    batchlen=0
    for batch,(words,tags) in enumerate(val.batches):
      output = model(words)
      output=output.view(-1,output.shape[2])
      output1=torch.nn.functional.softmax(output,dim=0).detach().numpy()
      answer=[np.argmax(output1[i]) for i in range(len(output1))]
      tags=tags.view(-1)
      actual=actual+tags.tolist()
      predict=predict + answer
      loss = criterion(output, tags)
      val_loss += loss.item()
      batchlen+=1
    answer=0
    for x in range(len(predict)):
      if predict[x]==actual[x]:
        answer=answer+1
    answer=answer/float(len(predict))
    # print(answer/float(len(predict)),len(predict),answer)
    
    print(f'Epoch {i+1} \t Training Loss: {loss_for_epoch} \t Validation Loss: {val_loss/batchlen} \t Accuray of validation :{answer} ')
    # print(classification_report(predict,actual))

    
  print("done")


def testdata(traindata,filename):
  sentences=[]
  Postags=[]
  with open(filename, 'r', encoding='utf-8') as file:
      sent=[]
      tags=[]
      index=-1
      for line in file:
        if not line.startswith('#'):
          line = line.strip()
          if line:
            fields = line.split('\t')
            ind=fields[0]
            if(index>int(ind)):
              sentences.append(sent)
              Postags.append(tags)
              sent=[]
              tags=[]
              
            else :
              sent.append(fields[1])
              tags.append(fields[3])
              # words.append([fields[1]])
              # Tags.append([fields[3]])
            index=int(ind)

  sentences.append(sent)
  Postags.append(tags)   
  length=len(sentences)
  sentwords=[]
  senttags=[]
  for i in range(0,length):
    length1=len(sentences[i])
    word_index=[]
    tag_index=[]
    for j in range(0,length1):
        word_index.append(traindata.vocabulary[sentences[i][j]])
        tag_index.append(traindata.POStags[Postags[i][j]])
    
    sentwords.append(word_index)
    senttags.append(tag_index)

  return sentwords,senttags
      



def Evaluation(model,data):
  # setences,tags=testdata(traindata,'/home/dell/NLP/ass2/test.conllu')
  actual=[]
  predict=[]
  # # print(setences[2])
  # for x in range(len(setences)):
  #   output=traindata(tensor(setences[x]))
  #   output=torch.nn.functional.softmax(output,dim=0).detach().numpy()
  #   # print(output)
  #   output=[np.argmax(output[i]) for i in range(len(setences[x]))]
  #   actual=actual + tags[x]
  #   predict=predict + output
  # print(output)
  for batch,(words,tags) in enumerate(data.batches):
      output = model(words)
      output=output.view(-1,output.shape[2])
      output=torch.nn.functional.softmax(output,dim=0).detach().numpy()
      # print(output.shape)
      answer=[np.argmax(output[i]) for i in range(len(output))]
      # print(answer)
      # print("\n")
      tags=tags.view(-1)
      # print(tags.shape)
      actual=actual+tags.tolist()
      predict=predict + answer
  answer=0
  print(classification_report(predict,actual,zero_division=0))



traindata=TagsAndWords('/home/dell/NLP/Assignment2/Assignment2/train.conllu')
Valdata=ValTags('/home/dell/NLP/Assignment2/Assignment2/dev.conllu',traindata)
# testdata=ValTags('/home/dell/NLP/ass2/test.conllu',traindata)
# print(Valdata.maxlength,testdata.maxlength)

# print(len(traindata.POStags))
# Trained_model=POSMODEL(len(traindata.vocabulary),len(traindata.POStags))
# training(Valdata,traindata,0.01,2)
# torch.save(traindata.state_dict(),'/home/dell/NLP/ass2/rrepochs.pth')
traindata.load_state_dict(torch.load('/home/dell/NLP/Assignment2/Assignment2/3epochs.pth'))
# Evaluation(traindata,Valdata)
print("Model loaded")

while(1):
  print("Give an input sentence")
  sentence=input()
  words_index=[]
  words = re.findall(r'\b\w+\b', sentence)
  print(words)
  sentence=words
  words_index=[traindata.vocabulary[word] for word in sentence]
  output=traindata(tensor(words_index))
  output=torch.nn.functional.softmax(output,dim=0).detach().numpy()
  output=[np.argmax(output[i]) for i in range(len(sentence))]

  for i in range(len(sentence)):
    for tag in traindata.POStags.get_stoi():
      if traindata.POStags[tag]==output[i]:
        print(f"{sentence[i]}\t {tag}")
        break

  

