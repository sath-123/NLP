import json
import numpy as np
import re
from scipy.linalg import svd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_distances
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy

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



def makevocab(data):
  single_word={}
  vocabulary={}
  for line in data:
    words=tokenization(cleaning(line))
    for word in words:
      if single_word.get(word)!=None:
        single_word[word]+=1
      else :
        single_word[word]=1
       
  length=len(words)
  vocabindex=0
  for key in single_word.keys():
    if single_word[key]>3:
      vocabulary[key]=vocabindex
      vocabindex+=1
    
  vocabulary['<unk>']=vocabindex
  return vocabulary


def wordTI(word_to_index,word):
  if word_to_index.get(word)==None:
    return word_to_index.get('<unk>')
  return word_to_index.get(word)


def co_matrix(data,vocabulary,wsize):
  co_occurrence = np.zeros((len(vocabulary), len(vocabulary)))
  size=int(wsize/2)
  for sentence in data:
    tokens=tokenization(cleaning(sentence))
    for i ,word in enumerate(tokens):
      for j in range(max(0, i-size), min(len(tokens), i +size+1)):
              if j != i:
                  co_occurrence[wordTI(vocabulary,word),wordTI(vocabulary,tokens[j])] += 1
                  
  return co_occurrence


def SVD(matrix,k):
  svd=TruncatedSVD(n_components=k)
  svd.fit(matrix)
  answer=svd.transform(matrix)
  return answer

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



	

# with open('sent.json', 'r') as f:
#     data = [json.loads(line)['text'] for line in f]

# vocabulary=makevocab(data)
# print(vocabulary['the'])
# matrix=co_matrix(data,vocabulary,4)
# reduced=SVD(matrix,20)


# np.savetxt('/content/drive/MyDrive/Ass3/model.csv',reduced )
# with open('//content/drive/MyDrive/Ass3/vocab.txt', 'w') as f:
#     json.dump(vocabulary, f)

with open('/home/dell/NLP/Ass3/vocab.txt', 'r') as f:
    vocabulary = json.load(f)

MODEL = np.loadtxt('/home/dell/NLP/Ass3/model.csv')

words=['wife','slowly','movies','titanic','looked']

anwer=Top_10('slowly',vocabulary,MODEL)
for x in anwer :
  print(x[0],x[1])



# for word in words:
#   result1=Top_10(word,vocabulary,MODEL)
#   dummy=[]
#   for x in result1:
#     w=x[0]
#     index=vocabulary[w]
#     word_vector=MODEL[index]
#     dummy.append(word_vector)
#   dummy=np.array(dummy)
#   D_2= TSNE(n_components=2,perplexity=5).fit_transform(dummy)
#   plt.scatter(D_2[:, 0], D_2[:, 1])
#   plt.annotate(word, xy=(D_2[0, 0], D_2[0, 1]), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')