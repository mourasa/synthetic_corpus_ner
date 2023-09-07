print("python train.py train.txt dev.txt test.txt model_name")

import sys
from flair.data import Corpus, Sentence
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from torch.utils.data import ConcatDataset

# 1. get the corpus
columns = {0: 'texts', 1: 'ner'}
corpus_folder = './corpus'
#corpus_train = ColumnCorpus(corpus_folder, columns, train_file=str(sys.argv[1]), dev_file='empty.txt', test_file='empty.txt')
corpus = ColumnCorpus(corpus_folder, columns, train_file=str(sys.argv[1]), dev_file=str(sys.argv[2]), test_file=str(sys.argv[3]))
#corpus = corpus.downsample(0.01)
print(corpus)

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_label_dictionary(label_type=tag_type)
print(tag_dictionary)

# 4. initialize each embedding we use
embedding_types = [

    # GloVe embeddings
    WordEmbeddings('fr'),

    # contextual string embeddings, forward
    FlairEmbeddings('fr-forward'),

    # contextual string embeddings, backward
    FlairEmbeddings('fr-backward'),
]

# embedding stack consists of Flair and GloVe embeddings
embeddings = StackedEmbeddings(embeddings=embedding_types)

# 5. initialize sequence tagger
from flair.models import SequenceTagger
from flair.models.word_tagger_model import WordTagger

tagger = SequenceTagger(embeddings=embeddings,
                    	tag_dictionary=tag_dictionary,
                    	tag_type=tag_type,
			loss_weights={'B-PER':1.0, 'I-PER':1.0, 'B-ORG':1.0, 'I-ORG':1.0, 'B-LOC':1.0, 'I-LOC':1.0, 'O':1.0},
			use_crf=True,
			tag_format="BIO")

#print(tagger.label_dictionary)


#prediction = tagger.predict([Sentence("Arthur est le premier roi de France.")], return_probabilities_for_all_classes=True, return_loss=True)
#print(prediction)

#tagger = SequenceTagger.load("./models/model_wiki/final-model.pt")

# 6. initialize trainer
from flair.trainers import ModelTrainer
import torch
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, BCELoss, CrossEntropyLoss

#tagger.loss_function = CrossEntropyLoss()

trainer = ModelTrainer(tagger, corpus)

# 7. run training
trainer.train('./models/'+str(sys.argv[4]),
              learning_rate=0.1,
              train_with_dev=True,
              mini_batch_size=50,
              max_epochs=50,
	      patience=3,
	      embeddings_storage_mode="cpu")


