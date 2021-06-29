

# torchtext训练(一) 概述与基本操作

## 一、概述

### 1、torchtext中的主要组件

#### torchtext主要包含的组件有：Field、Dataset 和 Iterator。

#### 1.1 Field

Field是用于**处理数据**的对象，处理的过程通过**参数指定**，且通过 Filed 能够构建 Example 对象。下面是定义 Field 对象的例子，

```python
TEXT = Field(sequential=True, tokenize=tokenize, lower=True, fix_length=200)
```

#### 1.2 Dataset

继承自 pytorch 的 Dataset，表示数据集。Dataset 可以看做是 Example 的实例集合；

#### 1.3 Iterator

Iterator 是 torchtext 到模型的输出，它提供了对数据的一般处理方式，比如打乱，排序，等等，可以动态修改 batch 大小

![image](https://github.com/liujiachi1997/liujiachi/blob/main/images1/20210627.png?raw=true)



## 二、QuickStart

```python
import pandas as pd
import torch
from torchtext import data
from torchtext.vocab import Vectors
from torchtext.legacy.data import TabularDataset, Dataset, BucketIterator, Iterator, Field
from torch.nn import init
from tqdm import tqdm
# Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)
```

### 2.1  展示数据格式

用到的数据集的是来自imdb电影评论数据集（.csv）文件，在 kaggle 的 imdb 官方页面下载：[link](https://www.kaggle.com/utathya/imdb-review-dataset/kernels)

我首先对数据集进行了切分，提取首行分别为‘train’ 和‘test’ 的数据并分成两个文件 train.csv、test.csv，展示个文件头5行的代码如下：

```python
df1 = pd.read_csv('./data/train.csv').head()
df2 = pd.read_csv('./data/test.csv').head()
print(df1)
print(df2)
```

output:

|      | id          | review                                            | label |
| ---- | ----------- | ------------------------------------------------- | ----- |
| 0    | 0_3.txt     | Story of a man who has unnatural feelings for ... | 0     |
| 1    | 10000_4.txt | Airport '77 starts as a brand new luxury 747 p... | 0     |
| 2    | 10001_4.txt | This film lacked something I couldn't put my f... | 0     |
| 3    | 10002_1.txt | Sorry everyone,,, I know this is supposed to b... | 0     |
| 4    | 10003_1.txt | When I was little my parents took me along to ... | 0     |
|      | id          | review                                            |
| ---- | ----------- | ------------------------------------------------- |
| 0    | 0_2.txt     | Once again Mr. Costner has dragged out a movie... |
| 1    | 10000_4.txt | This is an example of why the majority of acti... |
| 2    | 10001_1.txt | First of all I hate those moronic rappers, who... |
| 3    | 10002_3.txt | Not even the Beatles could write songs everyon... |
| 4    | 10003_3.txt | Brass pictures (movies is not a fitting word f... |

### 2.2  定义 Field

**Field**类定义了怎么处理不同类型数据（输入、标签）的方法，最主要的工作是指定好 **tokenize** 分词方法：

```python
tokenize = lambda x: x.split() # tokenize指定如何划分句子
# 匿名函数，传入 x,返回 x.split()
# 定义了两种Filed，分别用于处理文本和标签
TEXT = Field(sequential=True, tokenize=tokenize, lower=True, fix_length=200)
LABEL = Field(sequential=False, use_vocab=False)
```

### 2.3  构建Datasets

torchtext 集成了数据集方法 TabularDatasets（），可以直接处理数据集文件：

```python
# 3、构建 Dataset
fields = [("id", None),("review",TEXT),("label",LABEL)] # 列名与对应的Field对象
# TabularDataset：从csv、tsv、json的文件中读取数据并生成dataset
train, valid = TabularDataset.splits(				# 用.splits 分别生成
    path='data',
    train='train.csv',
    validation='valid.csv',
    format='csv',
    skip_header=True,
    fields=fields)

test_datafields = [('id', None),('review', TEXT)]

test = TabularDataset(								# 不用 splits ，单独生成
    path=r'data\test.csv',
    format='csv',
    skip_header=True,
    fields=test_datafields
)
print(type(train))
```

output:

```powershell
<class 'torchtext.legacy.data.dataset.TabularDataset'>
```

#### 构建词表 vocab

vocab 是一种 词 到 索引 的映射，我们在构建 one-hot 编码时，往往需要用到词在词典中的索引位置为参考，构建此表可以理解为 词向量化 的一部分：

```python
TEXT.build_vocab(train,valid,test,vectors='glove.6B.100d')
print(TEXT.vocab.stoi['<pad>'])			# stoi 是 str to index 的简称， <pad> 是补零符
print(TEXT.vocab.stoi['<unk>'])			# <unk> 是词典未出现词替代符
print(TEXT.vocab.itos[0])				# itos 是 index to str 的简称
print(TEXT.vocab.freqs.most_common(5))	# 最常出现的 5 个词语
print(vars(train.examples[0]))			# examples 是输入实例，vars() 函数返回对象object的属性和属性值的字典对象
```

注意到参数 vectors='glove.6B.100d' ，torchtext 集成了许多默认的词表，这里用到的是来自 glove 的 glove.6B.100d，此参数也可以是 vocab 文件的位置，输入文件名会自动下载词表并存储到 .vector_cache 文件夹下

output：

```powershell
1
0
<unk>
[('the', 1300651), ('a', 644690), ('and', 639355), ('of', 583691), ('to', 537848)]
{'review': ['story', 'of', 'a', 'man', 'who', 'has', 'unnatural', 'feelings', 'for', 'a', 'pig.', 'starts', 'out', 'with', 'a', 'opening', 'scene', 'that', 'is', 'a', 'terrific', 'example', 'of', 'absurd', 'comedy.', 'a', 'formal', 'orchestra', 'audience', 'is', 'turned', 'into', 'an', 'insane,', 'violent', 'mob', 'by', 'the', 'crazy', 
'chantings', 'of', "it's", 'singers.', 'unfortunately', 'it', 'stays', 'absurd', 'the', 'whole', 'time', 'with', 'no', 'general', 'narrative', 'eventually', 'making', 'it', 'just', 'too', 'off', 'putting.', 'even', 'those', 'from', 'the', 'era', 'should', 'be', 'turned', 'off.', 'the', 'cryptic', 'dialogue', 'would', 'make', 'shakespeare', 'seem', 'easy', 'to', 'a', 'third', 'grader.', 'on', 'a', 'technical', 'level', "it's", 'better', 'than', 'you', 'might', 'think', 'with', 'some', 'good', 'cinematography', 'by', 'future', 'great', 'vilmos', 'zsigmond.', 'future', 'stars', 'sally', 'kirkland', 'and', 'frederic', 'forrest', 'can', 'be', 'seen', 'briefly.'], 'label': 
'0'}
```

### 2.4  生成迭代器

声明训练、测试、验证过程中需要用到的迭代器，也是用自带的 BucketIterator 来做

```python
train_iter, valid_iter = BucketIterator.splits(
    (train, valid),
    batch_sizes=(8, 8),
    device='cpu',
    sort_key = lambda x: len(x.review),
    sort_within_batch = False,
    repeat = False
)

test_iter = Iterator(
    test,
    batch_size=8,
    device='cpu',
    sort = False,
    sort_within_batch= False,
    repeat = False
)
```

#### 调用迭代器

```python
for idx, batch in enumerate(train_iter):
    print(batch)
    print(batch.__dict__.keys())
    text, label = batch.review, batch.label
    print(text.shape, label.shape)
```

output:

```powershell
...
[torchtext.legacy.data.batch.Batch of size 8]
        [.review]:[torch.LongTensor of size 200x8]
        [.label]:[torch.LongTensor of size 8]
dict_keys(['batch_size', 'dataset', 'fields', 'input_fields', 'target_fields', 'review', 'label'])
torch.Size([200, 8]) torch.Size([8])

[torchtext.legacy.data.batch.Batch of size 8]
        [.review]:[torch.LongTensor of size 200x8]
        [.label]:[torch.LongTensor of size 8]
dict_keys(['batch_size', 'dataset', 'fields', 'input_fields', 'target_fields', 'review', 'label'])
torch.Size([200, 8]) torch.Size([8])

[torchtext.legacy.data.batch.Batch of size 8]
        [.review]:[torch.LongTensor of size 200x8]
        [.label]:[torch.LongTensor of size 8]
dict_keys(['batch_size', 'dataset', 'fields', 'input_fields', 'target_fields', 'review', 'label'])
torch.Size([200, 8]) torch.Size([8])

[torchtext.legacy.data.batch.Batch of size 8]
        [.review]:[torch.LongTensor of size 200x8]
        [.label]:[torch.LongTensor of size 8]
dict_keys(['batch_size', 'dataset', 'fields', 'input_fields', 'target_fields', 'review', 'label'])
torch.Size([200, 8]) torch.Size([8])

[torchtext.legacy.data.batch.Batch of size 8]
        [.review]:[torch.LongTensor of size 200x8]
        [.label]:[torch.LongTensor of size 8]
dict_keys(['batch_size', 'dataset', 'fields', 'input_fields', 'target_fields', 'review', 'label'])
torch.Size([200, 8]) torch.Size([8])

[torchtext.legacy.data.batch.Batch of size 8]
        [.review]:[torch.LongTensor of size 200x8]
        [.label]:[torch.LongTensor of size 8]
dict_keys(['batch_size', 'dataset', 'fields', 'input_fields', 'target_fields', 'review', 'label'])
torch.Size([200, 8]) torch.Size([8])

[torchtext.legacy.data.batch.Batch of size 8]
        [.review]:[torch.LongTensor of size 200x8]
        [.label]:[torch.LongTensor of size 8]
dict_keys(['batch_size', 'dataset', 'fields', 'input_fields', 'target_fields', 'review', 'label'])
torch.Size([200, 8]) torch.Size([8])

[torchtext.legacy.data.batch.Batch of size 8]
        [.review]:[torch.LongTensor of size 200x8]
        [.label]:[torch.LongTensor of size 8]
dict_keys(['batch_size', 'dataset', 'fields', 'input_fields', 'target_fields', 'review', 'label'])
torch.Size([200, 8]) torch.Size([8])
```





 



