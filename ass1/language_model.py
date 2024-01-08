import sys
import re
import random
import math

                                                         # creating list with n dictionaries,context list and cardinality list
ngrams=[]
ngrams_freq=[]
ngrams_car=[]
total_n=0


for x in range(0,5):
    ngrams.append({})
for x in range(0,4):
    ngrams_freq.append({})
for x in range(0,4):
    ngrams_car.append({})

                                                         # function for joining the words
def joining(data):
    length=len(data)
    if length==0:
        return ' '
    s=data[0]
    for x in range(1,length):
        s=s+' '+data[x]
    return s


 
                                                        # cleaning the data

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



                                                              #  fuction for tokenization
def tokenization(input):
    # input=re.sub(r'((<URL>)|(<Percent age>))',r' \1',input)
    # input=re.sub(r'(<(Percentage)>)',r'\1',input)
    input=re.sub(r'\((\w+)\)',r' \1',input)
    input=re.sub(r'([!-_.?^*\'{}\~/$`:"])',r' \1',input)
    tokens = input.split()
    # tokens=re.findall("<\w+>|\w+|[\.,\"\?\:\;']",input)
    return tokens

                                                       
                                                            # replacing words with frequency < 3 with UNK
def unknown_train(sentence,unigrams):                     
    cleaned=[]
    for word in sentence:
        if unigrams.get(word)<3:
            cleaned.append('<unk>')
        else :
            cleaned.append(word)
    return cleaned
    

continuation={}                                               # function for generating uni,bi,tri,four grams 
def generate_ngram(data):
    unigram ={}
    for sentence in data:
        for word in sentence:
            if word in unigram:
                unigram[word]+=1
            else:
                unigram[word]=1
    new_data=[]
    for sentence in data:
        new_data.append(unknown_train(sentence,unigram))

    for data in new_data:
        data = ['<s>']*3 + data +['.']+['</s>']*1              # padding with <s> and </s> to the list
        for order in range(1,5):
            length=len(data)-order+1
            for i in range(0,length):
                if(order!=1):
                    # print(order)
                    n_gram=data[i:i+order]
                    back_gram=data[i:i+order-1]
                    new_gram=joining(n_gram)
                    back_gram=joining(back_gram)
                    # print(back_gram)
                    # print(new_gram)
                    if new_gram in ngrams[order]:
                        ngrams[order][new_gram]+=1
                        ngrams_freq[order-1][back_gram]+=1
                    else:
                        ngrams[order][new_gram]=1
                        if ngrams_freq[order-1].get(back_gram)==None:
                            ngrams_freq[order-1][back_gram]=1
                            ngrams_car[order-1][back_gram]=1
                        else :
                            ngrams_freq[order-1][back_gram]+=1
                            ngrams_car[order-1][back_gram]+=1
                        
                        if order==2:
                            continuation[n_gram[1]]=continuation.get(n_gram[1],0)+1
                    
                else :
                    # print()
                    n_gram=data[i:i+order]
                    new_gram=joining(n_gram)
                    if new_gram in ngrams[order]:
                        ngrams[order][new_gram]+=1
                    else:
                        ngrams[order][new_gram]=1
                        # if order==2:
                        #     if n_gram[1] in bigrams:
                        #         bigrams[n_gram[1]]+=1
                        #     else:
                        #         bigrams[n_gram[1]]=1

    return sum(ngrams[1].values())


    


def atleastcout(n,ngrams,backoff_g):
    answer =0
    answer1=0
    # store ={}
    # backoff=backoff_g
    # for ngram in ngrams[n]:
    #     ngra=ngram.split(' ')
    #     ngra=ngra[:-1]
    #     ngra=joining(ngra) 
    #     if backoff==ngra:
    #         # if ngram not in store:
    #             answer+=1
    #             answer1+=ngrams[n][ngram]
    #     else :
    #             store[ngram]=1
    if ngrams_car[n-1].get(backoff_g)==None:
        answer=1e-5
        answer1=1e-5
        print("hy")
    else:
        answer=ngrams_car[n-1][backoff_g]
        answer1=ngrams_freq[n-1][backoff_g]
        # print(backoff_g)
        # print(answer1)


    
    if answer==0:
        print("who")
        answer= 1e-5
    if answer1==0:
        answer1=1e-5
    return answer,answer1




def kn_ney(data,n,i):
    cout1=0
    cout2=0
    cout3=0
    n_gram = joining(data[i+1-n:i+1])
    backoff_gram= joining(data[i+1-n:i])

    if n==1:
        num=continuation.get(n_gram,1e-5)
        # print(num,"jj")
        dem=len(ngrams[2])
        return num/dem
        
    if n!=1:
        if ngrams[n-1].get(backoff_gram)==None:
            cout2=1
            cout3=1e-5
            cout1=0.001
            # print(n_gram)
           
        else:
            cout3,total=atleastcout(n,ngrams,backoff_gram)
            cout2=total
            if n_gram in ngrams[n]:
               cout1=ngrams[n][n_gram]
            else :
                cout1=0.0001
                # print("hello")
            # cout2=ngrams[n-1][backoff_gram]
            # print(cout3,cout2,cout1)

    return (max(cout1-0.75,0)/cout2)+(0.75*cout3*kn_ney(data,n-1,i))/cout2



def witten_rec(data,n,i,total):
    cout1=0
    cout2=0
    cout3=0
    n_gram = joining(data[i+1-n:i+1])
    backoff_gram= joining(data[i+1-n:i])
    if n!=1:
        if ngrams[n-1].get(backoff_gram)==None:
            cout1=0
            cout3=1e-5
            # print(n_gram)
        else :
            # total=ngrams[n-1][backoff_gram]
            cout2,frea=atleastcout(n,ngrams,backoff_gram)
            cout3=cout2/(cout2+frea)
            if n_gram in ngrams[n]:
                cout1=ngrams[n][n_gram]/(frea+cout2)
            else:
                cout1=1e-5
                # print("kk")
        return (cout1+cout3*witten_rec(data,n-1,i,total))
    if n==1:
        if n_gram in ngrams[n]:
            cout1=ngrams[n][n_gram]
        else :
            cout1=1e-5
            # print("ll")
        # cout3=total_n
    return cout1/(sum(ngrams[n].values()))



def perplexity(sentence,method,total):
    length =len(sentence)
    p=1
    logs=0
    if method=='w':
        for i in range(3,length):
            logs=logs+math.log((witten_rec(sentence,4,i,total)))
        # if p==0:
        #     p=1
        return math.exp(logs*(-1/(length))),math.exp(logs)
    if method=='k':
        for i in range(3,length):
            logs=logs+math.log((kn_ney(sentence,4,i)))
            
            # print(p)
    # if p==0:
    #     p=1
    return math.exp(logs*(-1/float(length))),math.exp(logs)

    

def unknown_words(sentence):
    cleaned=[]
    for words in sentence:
        if words not in ngrams[1]:
            cleaned.append('<unk>')
        else :
            cleaned.append(words)
    return cleaned


def generating_output():
    # print()
   
    def files(path,file_train,file_test,method):
        with open(path, 'r') as f:
          inp = f.read()
        inp=cleaning(inp)
        inp=inp.split('.')
        data=[]
        for line in inp:
            data.append(tokenization(line))
                                                                        # splitting into training and testing
        random.shuffle(data)
        Train_data=data[1000:]
        Test_data=data[-1000:]
        print(len(Train_data),len(Test_data,),len(data))
        total_n=generate_ngram(Train_data)
        LM_train = open(file_train,'w+')
        LM_test = open(file_test,'w+')  
        avr_per=0
        sending_to_file=''
        for sentence in Train_data:
            # sen=sentence+['.']*1
            length=len(sentence)
            # for ran in range(length,4):
            #     sentence.append('<unk>')
            # length=len(sentence)
            # if length!=0:
            #    sentence[length-1]=sentence[length-1]+'.'
            sen=sentence
            sentence=['<s>']*3 + sentence+['.']+['</s>']*1
            # print("u")
            sentence=unknown_words(sentence)
            pep,p=perplexity(sentence,method,total_n)
            avr_per+=pep
            sen=joining(sen)
            pep=str(pep)
            # LM_train.write(sen+' '+pep)
            # LM_train.write("\n")
            sending_to_file = sending_to_file+sen+'.'+' '+pep+"\n"
        avr_per=avr_per/len(Train_data)
        avr_per=str(avr_per)
        LM_train.write(avr_per)
        LM_train.write('\n')
        LM_train.write(sending_to_file)
        print(avr_per)

        avr_per=0
        sending_to_file=''
        for sentence in Test_data:
            # print("s")
            # sen=sentence+['.']*1
            length=len(sentence)
            for ran in range(length,4):
                sentence.append('<unk>')
            length=len(sentence)
            # if length!=0:
            #    sentence[length-1]=sentence[length-1]+'.'
            sen=sentence
            sentence=['<s>']*3 + sentence+['.']+['</s>']*1
            sentence=unknown_words(sentence)
            pep,p=perplexity(sentence,method,total_n)
            avr_per+=pep
            sen=joining(sen)
            pep=str(pep)
            p=str(p)
            sending_to_file = sending_to_file+sen+'.'+' '+pep+"\n"
            # LM_test.write(sen+' '+pep)
            # LM_test.write("\n")
        avr_per=avr_per/len(Test_data)
        avr_per=str(avr_per)
        LM_test.write(avr_per)
        LM_test.write('\n')
        LM_test.write(sending_to_file)
        print(avr_per)
              
    path='/home/dell/NLP/ass1/b1.txt'
    file_train='/home/dell/NLP/ass1/2020101087_LM1_train-perplexity.txt'
    file_test='/home/dell/NLP/ass1/2020101087_LM1_test-perplexity.txt'
    files(path,file_train,file_test,'k')
    path='/home/dell/NLP/ass1/b1.txt'
    file_train='/home/dell/NLP/ass1/2020101087_LM2_train-perplexity.txt'
    file_test='/home/dell/NLP/ass1/2020101087_LM2_test-perplexity.txt'
    files(path,file_train,file_test,'w')
    # path='/home/dell/NLP/ass1/a1.txt'
    # file_train='/home/dell/NLP/ass1/2020101087_LM3_train-perplexity.txt'
    # file_test='/home/dell/NLP/ass1/2020101087_LM3_test-perplexity.txt'
    # files(path,file_train,file_test,'k')
    # path='/home/dell/NLP/ass1/a1.txt'
    # file_train='/home/dell/NLP/ass1/2020101087_LM4_train-perplexity.txt'
    # file_test='/home/dell/NLP/ass1/2020101087_LM4_test-perplexity.txt'
    # files(path,file_train,file_test,'w')

                                                                 ###################################################################
                                                                 #                     Input taking                                #
                                                                 ##################################################################
smoothing_type=sys.argv[1]
path = sys.argv[2]
# reading from the given text file
with open(path, 'r') as f:
    inp = f.read()



#tokenizing the data
inp=cleaning(inp)
inp=inp.split('.')
data=[]
for line in inp:
    data.append(tokenization(line))
                                                                 # splitting into training and testing
random.shuffle(data)
Train_data=data[1000:]
Test_data=data[-1000:]
# # print(len(Train_data))
# # print(len(Test_data))

# generating n grams

total_n=generate_ngram(Train_data)
print("training done")
while(1):
    print("Enter sentence : ")
    sentence = input()
    sentence=cleaning(sentence)
    sentence=tokenization(sentence)
    if len(sentence) <4:
        print("plzz enter valid sequence")
        continue
    
    sentence=['<s>']*3 + sentence+['</s>']*1
    sentence=unknown_words(sentence)
    print(perplexity(sentence,smoothing_type,total_n))
#################################
# generating_output() # generating input files
#################################










