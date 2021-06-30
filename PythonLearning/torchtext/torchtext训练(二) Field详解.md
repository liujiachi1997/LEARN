# torchtext训练(二)  Field 详解

## 1、数据处理过程

Field 是 torchtext 中最重要的对象，他基本定义了如何去处理数据

```python
from torchtext.legacy.data import Field,Example,Dataset, example
from torchtext import vocab
import os
import nltk
```

首先说明一下 Filed 对象的相关参数：

| 参数            | 说明                                                         |
| --------------- | ------------------------------------------------------------ |
| squential       | 数据是否为序列数据，默认为Ture。如果为False，则不能使用分词。 |
| use_vocab       | 是否使用词典，默认为True。如果为False，那么输入的数据类型必须是数值类型(即使用vocab转换后的)。 |
| init_token      | 文本的起始字符，默认为None。                                 |
| eos_token       | 文本的结束字符，默认为None。                                 |
| fix_length      | 所有样本的长度，不够则使用pad_token补全。默认为None，表示灵活长度。（**padding**） |
| tensor_type     | 把数据转换成的 tensor 类型 默认值为 torch.LongTensor。       |
| preprocessing   | 预处理pipeline， 用于分词 **tokenize** 之后、数值化 （vocab、vector） 之前，默认值为None。 |
| postprocessing  | 后处理pipeline，用于数值化之后、转换为tensor之前，默认为None。 |
| lower           | 是否把数据转换为小写，默认为False。                          |
| tokenize        | **分词**函数，默认为 str.split()                             |
| include_lengths | 是否返回一个已经补全的最小batch的元组和和一个包含每条数据长度的列表，默认值为 False。 |
| batch_first     | 是否用 batch 作为第一个维度                                  |
| pad_token       | 用于补全的字符，默认为 **<pad>** 。                          |
| unk_token       | 替换袋外词（vocab里面没有出现过的）的字符，默认为 **<unk>**。 |
| pad_first       | 是否从句子的开头进行补全（默认末尾补全），默认为 False。     |
| truncate_first  | 是否从句子的开头截断句子，默认为 False。                     |
| stop_words      | 停用词。                                                     |

接下来是 Filed 对象的应用举例：

```python
# 1.数据
corpus = ["D'aww! He matches this background colour",
         "Yo bitch Ja Rule is more succesful then",
         "If you have a look back at the source"]
labels = [0,1,0]
# 2.定义不同的Field
TEXT = Field(sequential=True, lower=True, fix_length=10,tokenize=str.split,batch_first=True)
LABEL = Field(sequential=False, use_vocab=False)
fields = [("comment", TEXT),("label",LABEL)]
# 3.将数据转换为Example对象的列表
exampels = []
for text, label in zip(corpus, labels):
    example = Example.fromlist(
        [text,label],
        fields=fields
    )
    exampels.append(example)
print(type(exampels[0]))
print(exampels[0].comment)
print(exampels[0].label)
# 4、构建词表
num_corpus = [example.comment for example in exampels]
# TEXT.build_vocab(num_corpus, vectors='glove.6B.100d')
TEXT.build_vocab(num_corpus)
print(TEXT.process(num_corpus))
```

output:

```powershell
<class 'torchtext.legacy.data.example.Example'>
["d'aww!", 'he', 'matches', 'this', 'background', 'colour']
0
tensor([[ 8, 10, 15, 22,  5,  7,  1,  1,  1,  1],
        [23,  6, 13, 17, 12, 16, 19, 21,  1,  1],
        [11, 24,  9,  2, 14,  4,  3, 20, 18,  1]])
```

如上所示，数据的处理过程大致分为两个阶段：**分词——向量化**



## 2、分词 Tokenize 与 词表 vocab

### 2.1  Tokenize

Field中的参数 tokenize 必须是一个函数，其作用是给定一个字符串，该函数以 **列表的形式** 返回分词结果。这里以 jieba 分词为例：

```python
# 二、Tokenize
# Field中的参数tokenize必须是一个函数，其作用是给定一个字符串，该函数以列表的形式返回分词结果。这里以jieba分词为例：
import jieba
# jieba分词返回的是迭代器，因此不能直接作为 tokenize
print(jieba.cut('我爱北京天安门'))
# 使用匿名函数 lambda 定义新的函数 cut，使其直接返回分词结果的列表，这样才可以作为 tokenize
cut = lambda x: list(jieba.cut(x))
print(cut('我爱北京天安门'))


# 下面来简单应用一下
corpusC = ['我爱北京天安门，天安门上太阳升',
           '伟大领袖毛主席，指引我们向前进！',
           '不忘初心，牢记使命']
labelC = [1, 1, 0]
# 定义 field
tokenize = cut
TEXT = Field(sequential=True, tokenize=tokenize, fix_length=10, batch_first=True)
LABEL = Field(sequential=False, use_vocab=False)
# 将数据转换为 Example 列表
fieldsC = [('comment', TEXT),('label', LABEL)]
exampelsC = []
for textC,labelC in zip(corpusC, labelC):
    exampelC = Example.fromlist(
        [textC, labelC],
        fields=fieldsC)
    exampelsC.append(exampelC)
print(type(exampelsC[0]))
print(exampelsC[0].comment)
print(exampelsC[0].label)
# 构建词表
new_corpusC = [exampelC.comment for exampelC in exampelsC]
TEXT.build_vocab(new_corpusC)    # 这里我们暂时不用中文 vocab
print(TEXT.process(new_corpusC))
```

output：

```powershell
['我', '爱', '北京', '天安门']
<class 'torchtext.legacy.data.example.Example'>
['我', '爱', '北京', '天安门', '，', '天安门', '上', '太阳升']
1
tensor([[13, 17, 10,  3,  2,  3,  4, 12,  1,  1],
        [ 6, 16,  2, 15, 14, 11,  9, 19,  1,  1],
        [ 5,  8,  2, 18,  7,  1,  1,  1,  1,  1]])
```

### 2.2 Vocab

前面的代码中 **Field 对象 TEXT** 通过调用 **build_vocab()** 方法来生成一个内置的Vocab对象，即  `TEXT.build_vocab(num_corpus)`  。下面看一下Vocab对象的常见用法：

```python
print(type(TEXT.vocab.freqs))   # freqs 是一个 Counter对象，包含了词表中单词的计数信息
print(TEXT.vocab.freqs['at'])
print(TEXT.vocab.itos[1])   # itos 表示 index to str
print(TEXT.vocab.stoi['<unk>'])   # itos 表示 str to index
print(TEXT.vocab.unk_index)
print(TEXT.vocab.vectors)   # 打印用到的词向量标准
```

output:

```powershell
<class 'collections.Counter'>
1
<pad>
0
0
None
```



## 3、向量化 Vectors

可以看到  `TEXT.vocab.vectors`  为None，因为在 build_vocab() 没有指定参数 vectors，此时可以通过 **load_vectors** 方法来加载词向量。load_vectors 的参数可以是字符串(例如: “fasttext.en.300d” )，其会**自动下载词向量并缓存到本地**。但如果是自己训练的词向量，则需要指定词向量在**本地的路径**。

### 3.1、自动下载并加载词向量

```python
TEXT.vocab.load_vectors('fasttext.en.300d')
print(TEXT.vocab.vectors.shape)
```

output:

```powershell
p = os.path.expanduser("~\\.vector_cache\\sgns.wiki.bigram-char")
TEXT.vocab.load_vectors(vocab.Vectors(p))
```

### 3.2、加载本地词向量

```python
#  注意：要放到 TEXT.build_vocab() 之后，否则报错
p = os.path.expanduser("~\\.vector_cache\\sgns.wiki.bigram-char")
TEXT.vocab.load_vectors(vocab.Vectors(p))
```



## 4、自定义 Field

接下来对 Field 对象进行一个优化（在实际做实验的时候我们可能经常要进行这种优化重写，以更适应于任务），主要实现以下两个功能：

1. 通过**字符串** (例如:“nltk”、“jieba”、“str”) 等方式**指定tokenize**
2. 能够通过**名称**来指定**自定义的词向量**（Field中build_vocab() 不能直接指定路径中的 vector 文件）

### 4.1  定义

```python
class MyField(Field):
    def __init__(self, tokenize = 'nltk', **kwargs):
        self.tokenize_name = tokenize
        tokenize = MyField._get_tokenizer(tokenize)
        super(MyField, self).__init__(tokenize=tokenize, **kwargs)  # 根据静态方法得到的 tokenizer 初始化

    @staticmethod   # 声明一个静态方法
    def _get_tokenizer(tokenizer):
        if tokenizer == 'nltk':
            return nltk.word_tokenize
        elif tokenizer == 'jieba':
            return lambda x : list(jieba.cut(x))
        elif tokenizer == 'split':
            return str.split
        else:
            raise ValueError('不支持的Tokenizer')   
    
    @classmethod    # 声明一个类方法
    def _get_vector_data(cls, vecs):
        if not isinstance(vecs, list):
            vecs = [vecs]
            # 先判断输入是否为列表，如果不是，转成列表
        
        vec_datas = []  # 词表化（向量化）后的数据放这里
        for vec in vecs:
            if not isinstance(vec, vocab.Vectors):
                if vec == 'glove.6B.50d':
                    embed_file = os.path.expanduser('.vector_cache\glove.6B.50d.txt')
                    vec_data = vocab.Vectors(embed_file)
                elif vec == 'glove.6B.100d':
                    embed_file = os.path.expanduser('.vector_cache\glove.6B.100d.txt')
                    vec_data = vocab.Vectors(embed_file)
                elif vec == 'glove.6B.200d':
                    embed_file = os.path.expanduser('.vector_cache\glove.6B.200d.txt')
                    vec_data = vocab.Vectors(embed_file)
                elif vec == 'glove.6B.300d':
                    embed_file = os.path.expanduser('.vector_cache\glove.6B.300d.txt')
                    vec_data = vocab.Vectors(embed_file)
                else:
                    raise ValueError('我只下载了 glove.6B 的词典，不好意思啊，不支持的向量类型。')
                vec_datas.append(vec_data)
            else:
                vec_datas.append(vec)
        return vec_datas
    
    def build_vocab(self, *args, vectors=None, **kwargs):
        if vectors is not None:
            vectors = MyField._get_vector_data(vectors)
        super(MyField, self).build_vocab(*args, vectors=vectors, **kwargs)
```

### 4.2  调用

```python
# 2.调用
corpus = ["D'aww! He matches this background colour",
         "Yo bitch Ja Rule is more succesful then",
         "If you have a look back at the source"]
labels = [0, 1, 0]
TEXT = MyField(sequential=True, lower=True, fix_length=10, tokenize='jieba', batch_first=True)
LABEL = MyField(sequential=False, use_vocab=False)
fields = [('comment', TEXT),('label', LABEL)]
examples = []
for text, label in zip(corpus, labels):
    example = Example.fromlist([text, label], fields=fields)
    examples.append(example)
data = [Dataset(example.comment, fields) for example in examples]
TEXT.build_vocab(data, vectors='glove.6B.300d')
print(len(TEXT.vocab.freqs))
print(TEXT.vocab.vectors.shape)
```

output:

```powershell
27
torch.Size([29, 300])
```

