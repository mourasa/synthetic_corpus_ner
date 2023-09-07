import sys
from flair.datasets import DataLoader, ColumnCorpus
from flair.models import SequenceTagger, WordTagger
from flair.data import Corpus

tagger: SequenceTagger = SequenceTagger.load('./models/'+str(sys.argv[1])+'/final-model.pt')

columns = {0: 'texts', 1: 'ner'}
corpus_folder = './corpus'
corpus: Corpus = ColumnCorpus(corpus_folder, columns, train_file='empty.txt', dev_file='empty.txt', test_file=str(sys.argv[2]))
print(corpus)

result = tagger.evaluate(corpus.test, gold_label_type="ner", out_path='./models/'+str(sys.argv[1])+'/predictions_isahit.txt')
print(result.detailed_results)
print(result.log_line)
#print(score)
#print(tagger._print_predictions(corpus.test, gold_label_type="ner"))
