## Link for models:
- https://drive.google.com/drive/folders/1jpFsByFHOKhjqrud-cxcMUgQ0ZEF2_Kk?usp=sharing

## 1. Preprocessing
- 1.1. loading dataset, train,test,val split
- 1.2. Tokenization, stemming, stopwords removal
- 1.3. wordtoind, padding, tensor, batches

## 2. ELMO Model for (input, next_predicted)
- 2.1. Pretrained Glove Embedding used, lstm , linear
- 2.2. training the model for 5 epochs
- 2.3. Adam optimizer, Cross Entropy loss

## 3. Save the model and load

- 4. downstream for (input,labels)
- 4.1 use embeddings, lstm1, lstm2 parameteres - concatenate, get output for number of classes , torch.max()
- 4.2  save and load the model

- 5. Evaluation
