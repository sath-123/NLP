import json
import numpy as np
import re
import random
from scipy.linalg import svd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_distances
from scipy import spatial
import torch
import torch.nn.functional as Fun
from torch import nn,optim
import math
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def cleaning(input):
    # print()
    input = input.lower()
    # URl
    input = re.sub(r'(https?://|www\.|http?://)\S+', 'URL', input)
    #hashtag
    input = re.sub(r'#\S+', 'Hashtag', input)
    #mention
    input = re.sub(r'@\S+', 'Mention', input)
    input = re.sub(r'[a-zA-Z0-9]*@\S+', 'Email', input)
    #percentage
    input = re.sub(r'\d+\%','Percentage',input)
    # date
    input=re.sub(r'\d{1,2}[-/.]\d{1,2}[/.-]\d{2,4}','Date',input)
    # replacing [ digit ] to place
    input=re.sub(r'\[\s*(\d+)\s*\]',' ',input)
    # removing all white spaces
    input=re.sub(r'\s+',' ',input)
    # Decimal
    input=re.sub(r'\d+\.\d+','Decimal',input)
    # number
    # input =re.sub(r'([ \b+ ])',' ',input)
    input=re.sub(r'\d+','Number',input)
    # replacing double punctuations
    input =re.sub(r'([{*.,-/&^%!_+=}])\1+',r'\1',input)
    #replacing words like _word_,_word,word_
    input=re.sub(r'_(\w+)_',r' \1',input)
    input=re.sub(r'_(\w+)',r' \1',input)
    input=re.sub(r'(\w+)_',r' \1',input)
    #replacing words like —word—,—word,word—
    input=re.sub(r'—\s+(\w+)\s+—',r' \1',input)
    input=re.sub(r'—(\w+)',r' \1',input)
    input=re.sub(r'(\w+)—',r' \1',input)
    #replacing words like -word-,-word,word-
    input=re.sub(r'-(\w+)-',r' \1',input)
    input=re.sub(r'-(\w+)',r' \1',input)
    input=re.sub(r'(\w+)-',r' \1',input)
    input=re.sub(r'mr\.','Mr',input)
    input=re.sub(r'mrs\.','Mrs',input)
    input=re.sub(r'r. w. (robert william)','Robert William',input)
    
    return input

def tokenization(input):
    # input=re.sub(r'((<URL>)|(<Percent age>))',r' \1',input)
    # input=re.sub(r'(<(Percentage)>)',r'\1',input)
    input=re.sub(r'\((\w+)\)',r' \1',input)
    input=re.sub(r'([!-_.?^*\'{}\~/$`:"])',r' \1',input)
    tokens = input.split()
    # tokens=re.findall("<\w+>|\w+|[\.,\"\?\:\;']",input)
    return tokens

def wordTI(word_to_index,word):
  if word_to_index.get(word)==None:
    return word_to_index.get('<unk>')
  return word_to_index.get(word)

def cosine_distance(vec1,vec2):
  return 1 - spatial.distance.cosine(vec1, vec2)

def Top_10(word,vocabulary,word_vectors):
  wordIndex=vocabulary[word]
  similarity={}
  for key in vocabulary.keys():
    index=vocabulary[key]
    # print(word_vectors[wordIndex])
    # print(wordIndex)
    similarity[key]=cosine_distance(word_vectors[wordIndex],word_vectors[index])
  final= sorted(similarity.items(), key=lambda x: x[1], reverse=True)
  return final[1:11]


class Word2Vec(nn.Module):
  def __init__(self,path):
    super().__init__()
    self.batch_size=64
    self.vocabulary={}
    self.emb_size=55
    self.window_size=4
    with open(path, 'r') as f:
      self.data = [json.loads(line)['text'] for line in f]
    self.vocabulary=self.makevocab()
    self.context,self.target,self.negative=self.output()
    self.bcontext,self.btarget,self.bnegative=self.make_batchs()
    self.embedding = nn.Embedding(len(self.vocabulary),self.emb_size,device=device)
    self.to(device)
    

  def makevocab(self):
    single_word={}
    vocabulary={}
    for line in self.data:
      words=tokenization(cleaning(line))
      for word in words:
        if single_word.get(word)!=None:
          single_word[word]+=1
        else :
          single_word[word]=1
        
    length=len(words)
    vocabindex=0
    for key in single_word.keys():
      if single_word[key]>5:
        vocabulary[key]=vocabindex
        vocabindex+=1
      
    vocabulary['<unk>']=vocabindex
    return vocabulary

  def negative_sampling(self):
    samples=[]
    for x in range(0,4):
      random_number = random.randint(0,100)
      samples.append(random_number)
    return samples
      
  def output(self):
    context=[]
    target=[]
    negative=[]
    size=int(self.window_size/2)
    for sentence in self.data:
      tokens=tokenization(cleaning(sentence))
      for i ,word in enumerate(tokens):
        dummy=[]
        for j in range(max(0, i-size), min(len(tokens), i +size+1)):
                if j != i:
                  dummy.append(wordTI(self.vocabulary,tokens[j]))

        if(len(dummy)==self.window_size):
          context.append(dummy)
          target.append(wordTI(self.vocabulary,word))
          negative.append(self.negative_sampling())
    return context,target,negative

  def make_batchs(self):
    self.target=torch.tensor(self.target,device=device)
    b_target=torch.split(self.target,self.batch_size)
    self.context=torch.tensor(self.context,device=device)
    b_context=torch.split(self.context,self.batch_size)
    self.negative=torch.tensor(self.negative,device=device)
    b_negative=torch.split(self.negative,self.batch_size)
    return b_context,b_target,b_negative


  def forward(self,context,target,negative):
    # print()
    context_e=self.embedding(context)
    target_e=self.embedding(target)
    negative_e=self.embedding(negative)
    context_e=context_e.mean(dim=1)
    # print(np.shape(target_e))
    positive=(context_e*target_e).sum(dim=1)
    # print(np.shape(context_e))
    positive_loss=Fun.logsigmoid(positive)
    positive_loss=positive_loss.mean()

    negative_e=negative_e.mean(dim=1)
    negative=(negative_e*target_e).sum(dim=1)
    negative_loss=Fun.logsigmoid(-negative)
    negative_loss=negative_loss.mean()
    return -(positive_loss+negative_loss)

def training(model,L_rate,epochs):
  # criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(),lr=L_rate)
  initial_loss=-math.inf
  model.train()
  # total_loss=0
  for i in range(epochs):
    loss_for_epoch=0
    print("entered")
    batchs=0
    for batch ,context in enumerate(model.bcontext):
      optimizer.zero_grad() # set the gradients to zero before starting to do backpropragation 
      loss=model.forward(context,model.btarget[batch],model.bnegative[batch])
      loss.backward()
      optimizer.step()
      loss_for_epoch+=loss.item()
      batchs+=1
    loss_for_epoch=loss_for_epoch/batchs
    print(loss_for_epoch)
    print(f'Epoch {i+1} \t\t Training Loss: {loss_for_epoch} ')
    
    
training(MODEL,0.01,5)

                           # code for training the data set

# MODEL=Word2Vec('/content/drive/MyDrive/ass3nlp/sent.json')
# word_vectors=MODEL.embedding.weight.cpu().detach().numpy()
# np.savetxt('/content/drive/MyDrive/ass3nlp/model2.csv',word_vectors )
# with open('/content/drive/MyDrive/ass3nlp/vocab2.txt', 'w') as f:
#     json.dump(MODEL.vocabulary, f)

                            # code for loading the dataset

with open('/content/drive/MyDrive/ass3nlp/vocab.txt', 'r') as f:
    vocabulary = json.load(f)
word_vectors = np.loadtxt('/content/drive/MyDrive/ass3nlp/model.csv')

words=['wife','slowly','movies','titanic','looked']
  
                            # code for plooting top 10 words for 5 words
for word in words:
  result1=Top_10(word,vocabulary,word_vectors)
  dummy=[]
  for x in result1:
    w=x[0]
    index=MODEL.vocabulary[w]
    word_vector=word_vectors[index]
    dummy.append(word_vector)
  dummy=np.array(dummy)
  D_2= TSNE(n_components=2,perplexity=5).fit_transform(dummy)
  plt.scatter(D_2[:, 0], D_2[:, 1])
  plt.annotate(word, xy=(D_2[0, 0], D_2[0, 1]), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')


                                          # code for finding top 10 words using gensim pretrained model
# import gensim.downloader as api
# # download the pre-trained word2vec model
# model = api.load('word2vec-google-news-300')
# # get the top 10 most similar word vectors for "titanic"
# similar_words = model.most_similar('titanic', topn=10)
# for word, similarity in similar_words:
#     vector = model[word]
#     print(f"{word}: {similarity}")
