# N-gram model with smoothing
# Main functions
* ## cleaning() for cleaning data
* ## tokenization() used for doing tokenization
* ## unkown_train() converts all low frequency words to <unk>
* ## generate_ngram() used for finding all ngrams also cardinalities and cout of context
* ## kn_ney() for kneser ney smoothing
* ## witten_rec() for witten-bell smoothing
* ## perplexity() for calculating perplexity
* ## unknown-words() changing unkwon words in input sequence to <unk>
* ## generating_output() for generating perplexity scores for all sentences in test and train set and finding average perplexity score


# How to run the file
* ## language_model.py <method> inputpath
* ## then it will ask to input sentence and it will return perplexity and probability of that sentence

# Schemes used for tokenization
* ## replaced -word-,-word,word- to word similarly _word_
* ## replaced [ digits ] to space
* ## replaced repeated punctuations to single !!!! ---> !

# NLM
## functions
* ## MODEL() to create model
* ## training() for training the model
* ## perplexity_for_sequence() for calculating perplexity for input sentence

## https://drive.google.com/drive/folders/1o5Wu0l-WhwR6_PsfHyq167SjxGAgcbKA?usp=sharing