print("python train.py percent_wiki model_name_folder")

import sys
from flair.data import Corpus
from flair.datasets.sequence_labeling import NER_MULTI_WIKINER
from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings

# 1. get the corpus
corpus = NER_MULTI_WIKINER("fr").downsample(float(sys.argv[1]))
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

# 6. initialize trainer
from flair.trainers import ModelTrainer

trainer = ModelTrainer(tagger, corpus)

# 7. run training
trainer.train('../models/'+str(sys.argv[2]),
              learning_rate=0.1,
              train_with_dev=True,
              mini_batch_size=32,
              max_epochs=50,
	      patience=3,
	      embeddings_storage_mode="gpu")

