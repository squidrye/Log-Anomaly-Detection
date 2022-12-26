# Leveraging Word2vec for Text Classification

Many machine learning algorithms requires the input features to be represented as a fixed-length feature vector. When it comes to texts, one of the most common fixed-length features is one hot encoding methods such as bag of words or tf-idf. The advantage of these approach is that they have fast execution time, while the main drawback is they lose the ordering & semantics of the words.

The motivation behind converting text into semantic vectors (such as the ones provided by Word2Vec) is that not only do these type of methods have the capabilities to extract the semantic relationships (e.g. the word powerful should be closely related to strong as oppose to another word like bank), but they should be preserve most of the relevant information about a text while having relatively low dimensionality.


# Gensim Implementation

After feeding the Word2Vec algorithm with our corpus, it will learn a vector representation for each word. This by itself, however, is still not enough to be used as features for text classification as each record in our data is a document not a word.

To extend these word vectors and generate document level vectors, we'll take the naive approach and use an average of all the words in the document

we build a pipeline that transforms the text into low dimensional vectors via average word vectors as use it to fit a boosted tree model, we then report the performance of the training/test set.

The transformers folder that contains the implementation is at the following link https://github.com/ethen8181/machine-learning/tree/master/keras/text_classification/transformers
