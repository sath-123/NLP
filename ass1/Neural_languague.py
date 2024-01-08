import torch
import math
import re
import random
from torch import nn,optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import sys
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
    input=re.sub(r'\[\s*(\d+)\s*\]',' ',input)
    input=re.sub(r'\d{1,2}[-/.]\d{1,2}[/.-]\d{2,4}','Date',input)
    # removing all white spaces
    input=re.sub(r'\s+',' ',input)
    # Decimal
    input=re.sub(r'\d+\.\d+','Decimal',input)
    # number
    # input =re.sub(r'([ \b+ ])',' ',input)
    input=re.sub(r'\d+','Number',input)
    # replacing double punctuations
    input =re.sub(r'([{*.,-/&^%!_+=}])\1+',r'\1',input)
    #
    input=re.sub(r'_(\w+)_',r' \1',input)
    input=re.sub(r'_(\w+)',r' \1',input)
    input=re.sub(r'(\w+)_',r' \1',input)

    input=re.sub(r'—\s+(\w+)\s+—',r' \1',input)
    input=re.sub(r'—(\w+)',r' \1',input)
    input=re.sub(r'(\w+)—',r' \1',input)

    input=re.sub(r'-(\w+)-',r' \1',input)
    input=re.sub(r'-(\w+)',r' \1',input)
    input=re.sub(r'(\w+)-',r' \1',input)
    

    
    # input=re.sub(r'—(\w+)—',r' \1',input)
    # input=re.sub(r'(\w+).',r' \1',input)
    # input=re.sub(r'_you_',r'you',input
    input=re.sub(r'mr\.','Mr',input)
    input=re.sub(r'mrs\.','Mrs',input)
    input=re.sub(r'r. w. (robert william)','Robert William',input)
    
    return input



# fuction for tokenization
def tokenization(input):
    # input=re.sub(r'((<URL>)|(<Percent age>))',r' \1',input)
    # input=re.sub(r'(<(Percentage)>)',r'\1',input)
    input=re.sub(r'\((\w+)\)',r' \1',input)
    input=re.sub(r'([!-_.?^*\'{}\~/$`:"])',r' \1',input)
    tokens = input.split()
    # tokens=re.findall("<\w+>|\w+|[\.,\"\?\:\;']",input)
    return tokens

class MODEL(nn.Module): 
  def __init__(self,traindata,batch_size):
    super().__init__()
    self.batch_size=batch_size
    self.prefix=[]
    self.maximumlen=0
    self.words=self.get_words(traindata)
    self.single_word={}
    self.vocab=self.unique_words()
    print(len(self.vocab))
    self.word_to_index= {word : i for i ,word in enumerate(self.vocab)}
    self.context_words=self.context()
    self.context_batch=self.batching()

    self.hidden_size=130
    self.emb_dim=100
    self.embedding = nn.Embedding(len(self.vocab),self.emb_dim,device=device)
    self.lstm = nn.LSTM(input_size=self.emb_dim, hidden_size=self.hidden_size)
    self.answer = nn.Linear(self.hidden_size,len(self.vocab),device=device)
    self.to(device)
    
  def get_words(self,traindata):
    # it will give all words of the traindata
    vocab=[]
    maxlen=0
    for line in traindata:
      tokens=tokenization(line)
      length=len(tokens)
      for i in range(1,length):
        self.prefix.append(tokens[:i+1])
      if(maxlen<len(tokens)):
        maxlen=len(tokens)
      vocab = vocab + tokens
    self.maximumlen=maxlen
    return vocab

  def unique_words(self):
    # for geeting unique words
    for word in self.words:
      if self.single_word.get(word)!=None:
        self.single_word[word]+=1
      else :
        self.single_word[word]=1
    length=len(self.words)
    for i in range(0,length):
      if self.single_word[self.words[i]]< 3:
        self.words[i]='<unk>'
    # self.words.append('<unk>')
    u_words=list(set(self.words))
    u_words.append('<unk>')
    u_words=list(set(u_words))

    u_words.append('<pad>')
    return u_words


  def context(self):
    # converting prefix sequences to equal length and from word to index
    i=0
    for prefix_sequence in self.prefix:
      length =len(prefix_sequence)
      word_index=[]
      for word in prefix_sequence:
        if self.single_word[word]<3:
          word_index.append(self.word_to_index['<unk>'])
        else :
          word_index.append(self.word_to_index[word])
      self.prefix[i]=[self.word_to_index['<pad>']]*(self.maximumlen-length) + word_index
      i+=1

  def batching(self):
    # dividing the prefix sequences into batches of batch_size
     self.prefix=torch.tensor(self.prefix,device=device)
     batch_wise=torch.split(self.prefix,self.batch_size)
     batchs=[]
     for batch in batch_wise:
       batchs.append((batch[:,:-1],batch[:,-1]))
     return batchs
  
  def forward(self,context):
    # for lstm model
    embed_output=self.embedding(context)
    out,state=self.lstm(embed_output)
    final=self.answer(out[:,-1])
    return final

def training(model,L_rate,epochs):
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(),lr=L_rate)
  initial_loss=-math.inf
  for i in range(epochs):
    loss_for_epoch=0
    print("entered")
    batchs=0
    for batch ,(history,prediction) in enumerate(model.context_batch):
      optimizer.zero_grad() # set the gradients to zero before starting to do backpropragation 
      predicted_one=model.forward(history)
      loss=criterion(predicted_one,prediction)
      loss.backward()
      optimizer.step()
      loss_for_epoch+=loss.item()
      batchs+=1
    loss_for_epoch=loss_for_epoch/batchs
    if abs(initial_loss-loss_for_epoch)<=0.001:
      break
    initial_loss=loss_for_epoch
    
  print("done")
def perplexity_for_sentence(model,text):
  # print(model.word_to_index['The'],model.word_to_index['<unk>'])
  text=cleaning(text)
  tokens=tokenization(text)
  # sen=tokens

  length=len(tokens)
  if length > model.maximumlen or length<2:
    return 0,0,1
  # print(tokens)
  # print(model.word_to_index['the'],model.word_to_index['<unk>'])
  # print(length)
  for i in range(length):
    if tokens[i] not in model.vocab:
      tokens[i]='<unk>'
    
  for i in range(length):
    # print(tokens[i])
    tokens[i]=model.word_to_index[tokens[i]]
  prefix_sequences=[]
  for i in range(1,length):
    prefix_sequences.append(tokens[:i+1])
  s=0
  for sequence in prefix_sequences:
    siz=len(sequence)
    prefix_sequences[s]=(model.maximumlen-siz)*[model.word_to_index['<pad>']]+prefix_sequences[s]
    s+=1
  # predicted=prefix_sequences[:,-1]
  prefix_sequences=torch.tensor(prefix_sequences,device=device)
  words=prefix_sequences[:,-1]
  context=prefix_sequences[:,:-1]
  distribution=model.forward(context)
  # print(distribution[0])
  p = torch.nn.functional.softmax(distribution, dim=0).cpu().detach().numpy()
  # print(distribution[0][1].item())
  prob=1
  for i in range(0,s):
    # print(words[i])
    # num=model.word_to_index[words[i]]
    probs=p[i][words[i]]
    prob=prob*probs
    # print(probs.item())
  if s==0 or prob==0:
    return 0,0,1

  return prob**(-1/s),prob,0

def perplexity_for_test(model,testfile,outputfile):
  m=0
  f=open(testfile,'r')
  inp=f.read()
  inp=inp.split('\n')
  # print(m,"kk")
  fileout=open(outputfile,'w+')
  answer=''
  average=0
  s=0
  for line in inp:
    pep,p,test=perplexity_for_sentence(model,line)
    if test==0:
      answer=answer+line+'.'+' '+str(pep)+'\n'
      average=average+pep
      s+=1
    
  # print(s,"kkkk")
  average=average/s
  fileout.write(str(average))
  fileout.write('\n')
  fileout.write(answer)




path=sys.argv[1]

################################for split##########
# with open(path, 'r') as f:
#     inp = f.read()
# inp=cleaning(inp)
# inp=list(filter(None,inp.split('.')))
# # inp=inp.split('.')
# print(len(inp))
# data=[]
# lines=0
# for line in inp:
#   lines+=1
#   data.append(line)
#########################################
f=open(path,'r')
data=[]
lines=0
for line in f.readlines():
  lines+=1
  data.append(line)
print(lines)

###################################for spliting test,train,validation########
# trainlength=int(lines*70/100)
# testlength=int(lines*15/100)
# print(trainlength,testlength)
# # 3800,4613,-813
# random.shuffle(data)
# Train_data=data[:15145]
# Val_data=data[15145:18390]
# Test_data=data[-3245:]
# print(len(data),len(Train_data),len(Val_data),len(Test_data))
# trainfile = open('/content/drive/MyDrive/nlp/file2_train.txt','w+')
# testfile=open('/content/drive/MyDrive/nlp/file2_test.txt','w+')
# valfile=open('/content/drive/MyDrive/nlp/file2_val.txt','w+')
# spr=0
# for x in Train_data:
#   spr+=1
#   trainfile.write(x+'\n')
#   # trainfile.write("\n")
# for x in Test_data:
#   testfile.write(x)
#   testfile.write("\n")
# for x in Val_data:
#   valfile.write(x)
#   valfile.write("\n")


###############################################end###################
Trained_model=MODEL(data,256)
print(Trained_model.prefix[1][Trained_model.maximumlen-2])
# training(Trained_model,0.001,15)
# torch.save(Trained_model.state_dict(),'/content/drive/MyDrive/nlp/eng_lm15model2.pth')
Trained_model.load_state_dict(torch.load('/home/dell/NLP/ass1/end_lm12model1.pth'))
# perplexity_for_test(Trained_model,'/content/drive/MyDrive/nlp/file1_train.txt','/content/drive/MyDrive/nlp/train.txt')

#print(perplexity_for_sentence(Trained_model,'console lady catherine as well as you can'))



