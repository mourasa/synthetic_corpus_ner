from flair.datasets import DataLoader, ColumnCorpus
from flair.models import SequenceTagger
from flair.data import Corpus

tagger: SequenceTagger = SequenceTagger.load('../models/model_wiki_and_synth/model_wand_746/final-model.pt')

columns = {0: 'texts', 1: 'ner'}
corpus_folder = '../corpus/corpus_synth'
corpus: Corpus = ColumnCorpus(corpus_folder, columns, train_file='empty.txt', dev_file='empty.txt', test_file='corpus_std1.txt')
print(corpus)

result = tagger.evaluate(corpus.test, mini_batch_size=32, out_path=f"../corpus/corpus_synth/predictions_wiki_and_synth.txt", gold_label_type="ner")
print(result.detailed_results)
