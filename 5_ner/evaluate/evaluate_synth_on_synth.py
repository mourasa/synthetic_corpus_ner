from flair.datasets import DataLoader, ColumnCorpus
from flair.models import SequenceTagger
from flair.data import Corpus

tagger: SequenceTagger = SequenceTagger.load('./models/model_synth/final-model.pt')

columns = {0: 'texts', 1: 'ner'}
corpus_folder = './corpus/corpus_synth'
corpus: Corpus = ColumnCorpus(corpus_folder, columns, train_file='empty.txt', dev_file='empty.txt', test_file='corpus.txt')
print(corpus)

result = tagger.evaluate(corpus.test, mini_batch_size=32, out_path=f"./corpus/corpus_synth/predictions.txt", gold_label_type="ner")
print(result.detailed_results)
