import sys
from flair.datasets import DataLoader, ColumnCorpus
from flair.models import SequenceTagger
from flair.data import Corpus

tagger: SequenceTagger = SequenceTagger.load('./models/'+str(sys.argv[1])+'/final-model.pt')

columns = {0: 'texts', 1: 'ner'}
corpus_folder = './corpus'
corpus: Corpus = ColumnCorpus(corpus_folder, columns, train_file='empty.txt', dev_file='empty.txt', test_file=str(sys.argv[2]))
print(corpus)

result = tagger.evaluate(corpus.test, mini_batch_size=32, out_path=f"./corpus/"+str(sys.argv[3]), gold_label_type="ner")
print(result.detailed_results)
