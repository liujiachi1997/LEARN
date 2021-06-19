# TORCHTEXT.DATA.UTILS

使用 torchtext 自带的数据集的一般方法：

```python
# import datasets
from torchtext.datasets import IMDB

train_iter = IMDB(split='train')

def tokenize(label, line):
    return line.split()

tokens = []
for label, line in train_iter:
    tokens += tokenize(label, line)
```

可以用的数据集：

Datasets

- [Text Classification](https://pytorch.org/text/stable/datasets.html#text-classification)
  - [AG_NEWS](https://pytorch.org/text/stable/datasets.html#ag-news)
  - [SogouNews](https://pytorch.org/text/stable/datasets.html#sogounews)
  - [DBpedia](https://pytorch.org/text/stable/datasets.html#dbpedia)
  - [YelpReviewPolarity](https://pytorch.org/text/stable/datasets.html#yelpreviewpolarity)
  - [YelpReviewFull](https://pytorch.org/text/stable/datasets.html#yelpreviewfull)
  - [YahooAnswers](https://pytorch.org/text/stable/datasets.html#yahooanswers)
  - [AmazonReviewPolarity](https://pytorch.org/text/stable/datasets.html#amazonreviewpolarity)
  - [AmazonReviewFull](https://pytorch.org/text/stable/datasets.html#amazonreviewfull)
  - [IMDb](https://pytorch.org/text/stable/datasets.html#imdb)
- [Language Modeling](https://pytorch.org/text/stable/datasets.html#language-modeling)
  - [WikiText-2](https://pytorch.org/text/stable/datasets.html#wikitext-2)
  - [WikiText103](https://pytorch.org/text/stable/datasets.html#wikitext103)
  - [PennTreebank](https://pytorch.org/text/stable/datasets.html#penntreebank)
- [Machine Translation](https://pytorch.org/text/stable/datasets.html#machine-translation)
  - [Multi30k](https://pytorch.org/text/stable/datasets.html#multi30k)
  - [IWSLT2016](https://pytorch.org/text/stable/datasets.html#iwslt2016)
  - [IWSLT2017](https://pytorch.org/text/stable/datasets.html#iwslt2017)
- [Sequence Tagging](https://pytorch.org/text/stable/datasets.html#sequence-tagging)
  - [UDPOS](https://pytorch.org/text/stable/datasets.html#udpos)
  - [CoNLL2000Chunking](https://pytorch.org/text/stable/datasets.html#conll2000chunking)
- [Question Answer](https://pytorch.org/text/stable/datasets.html#question-answer)
  - [SQuAD 1.0](https://pytorch.org/text/stable/datasets.html#squad-1-0)
  - [SQuAD 2.0](https://pytorch.org/text/stable/datasets.html#squad-2-0)
- [Unsupervised Learning](https://pytorch.org/text/stable/datasets.html#unsupervised-learning)
  - [EnWik9](https://pytorch.org/text/stable/datasets.html#enwik9)

