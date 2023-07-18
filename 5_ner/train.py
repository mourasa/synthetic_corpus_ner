print("python train.py train.txt dev.txt test.txt model_name")

import sys
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings

# 1. get the corpus
columns = {0: 'texts', 1: 'ner'}
corpus_folder = './corpus'
corpus: Corpus = ColumnCorpus(corpus_folder, columns, train_file=str(sys.argv[1]), dev_file=str(sys.argv[2]), test_file=str(sys.argv[3]))
corpus = corpus.downsample(0.01)
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

tagger = SequenceTagger(hidden_size=256,
                        embeddings=embeddings,
                        tag_dictionary=tag_dictionary,
                        tag_type=tag_type)
#tagger = SequenceTagger.load("./models/model_wiki/final-model.pt")

# 6. initialize trainer
from flair.trainers import ModelTrainer

trainer = ModelTrainer(tagger, corpus)

# 7. run training
trainer.train('./models/'+str(sys.argv[4]),
              learning_rate=0.1,
              train_with_dev=True,
              mini_batch_size=32,
              max_epochs=50,
	      patience=3,
	      embeddings_storage_mode="gpu")

