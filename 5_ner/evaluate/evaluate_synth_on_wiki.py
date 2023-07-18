from flair.datasets import DataLoader, ColumnCorpus
from flair.models import SequenceTagger
from flair.data import Corpus
from flair.datasets.sequence_labeling import NER_MULTI_WIKINER

tagger: SequenceTagger = SequenceTagger.load('./models/model_synth/final-model.pt')

corpus: Corpus = NER_MULTI_WIKINER("fr")
print(corpus)

result = tagger.evaluate(corpus.test, mini_batch_size=32, out_path=f"./corpus/corpus_wiki/predictions_synth.txt", gold_label_type="ner")
print(result.detailed_results)
