print("python train.py percent_wiki train.txt dev.txt test.txt model_name_folder")

import sys
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.datasets.sequence_labeling import NER_MULTI_WIKINER
from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from torch.utils.data import ConcatDataset

# 1. get the corpus
columns = {0: 'texts', 1: 'ner'}
corpus_folder = '../corpus/corpus_synth'
corpus: Corpus = ColumnCorpus(corpus_folder, columns, train_file=str(sys.argv[2]), dev_file=str(sys.argv[3]), test_file=str(sys.argv[4]))
print(corpus)

# 2. what tag do we want to predict?
tag_type = 'ner'
# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_label_dictionary(label_type=tag_type, add_unk=True)
print(tag_dictionary)

corpus_wiki: Corpus = NER_MULTI_WIKINER("fr").downsample(float(sys.argv[1]))
print(corpus_wiki)
corpus_wiki.label_dictionary = tag_dictionary
corpus.label_dictionary = tag_dictionary

corpus_train = ConcatDataset([corpus_wiki._train,corpus._train])

corpus._train = corpus_train

print(corpus)

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
trainer.train('../models/model_wiki_and_synth/'+str(sys.argv[5]),
              learning_rate=0.1,
              train_with_dev=True,
              mini_batch_size=32,
              max_epochs=50,
	      patience=3,
	      embeddings_storage_mode="cpu")

