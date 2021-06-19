# Torchtext 学习笔记——数据功能（TORCHTEXT.DATA.FUNCTIONAL）

## generate_sp_model （生成sp模型）

```python
torchtext.data.functional.generate_sp_model(filename, vocab_size=20000, model_type='unigram', model_prefix='m_user')
```

用途：

​		训练 SentencePiece 分词器 (tokenizer)

参数如下：

- **filename** – 用于训练分词器的数据文件
- **vocab_size** – 词汇量（默认：20,000）
- **model_type** – SentencePiece模型的类型，包括unigram、bpe、char、word
- **model_prefix** – 保存模型和词汇的文件的前缀（prefix）

输出：

​		模型和词汇以 model_prefix 保存在两个单独的文件中。

示例：

```python
>>> from torchtext.data.functional import generate_sp_model
>>> generate_sp_model('test.csv', vocab_size=23456, model_prefix='spm_user')
```



## load_sp_model （加载sp模型）

```python
torchtext.data.functional.load_sp_model(spm)
```

用途：

​		为文件加载 sentencepiece 模型

参数：

* **spm** – the file path or a file object saving the sentencepiece model.  保存句子模型的文件路径或文件对象。

输出：

​		一个 SentencePiece 模型

示例：

```python
>>> from torchtext.data.functional import load_sp_model
>>> sp_model = load_sp_model("m_user.model")
>>> sp_model = load_sp_model(open("m_user.model", 'rb'))
```



## sentencepiece_numericalizer （基于模型的数字化器）

```python
torchtext.data.functional.sentencepiece_numericalizer(sp_model)
```

用途：

​		将文本句子数字化为句子模型 id 上的生成器。

参数：

* **sp_model** – 一个 SentencePiece 模型

输出：

​		基于 SentencePiece 模型的具有文本句子输入和相应 id 输出的生成器。

示例：

```python
>>> from torchtext.data.functional import sentencepiece_numericalizer
>>> sp_id_generator = sentencepiece_numericalizer(sp_model)
>>> list_a = ["sentencepiece encode as pieces", "examples to   try!"]
>>> list(sp_id_generator(list_a))
    [[9858, 9249, 1629, 1305, 1809, 53, 842],
     [2347, 13, 9, 150, 37]]
```



## sentencepiece_tokenizer （基于模型的令牌转化器）

```python
torchtext.data.functional.sentencepiece_tokenizer(sp_model)
```

用途：

​		一个 sentencepiece 模型，用于将句子 令牌化 为基于令牌的生成器

参数：

* **sp_model** – 一个 SentencePiece 模型

输出：

​		基于 SentencePiece 模型的具有文本句子输入和相应标记输出的生成器。

示例：

```python
>>> from torchtext.data.functional import sentencepiece_tokenizer
>>> sp_tokens_generator = sentencepiece_tokenizer(sp_model)
>>> list_a = ["sentencepiece encode as pieces", "examples to   try!"]
>>> list(sp_tokens_generator(list_a))
    [['_sentence', 'piece', '_en', 'co', 'de', '_as', '_pieces'],
     ['_example', 's', '_to', '_try', '!']]
```



## custom_replace (替换方法)

```python
torchtext.data.functional.custom_replace(replace_pattern)
```

用途：

​		一种文本字符串转换方法

示例：

```python
>>> from torchtext.data.functional import custom_replace
>>> custom_replace_transform = custom_replace([(r'S', 's'), (r'\s+', ' ')])
>>> list_a = ["Sentencepiece encode  aS  pieces", "exampleS to   try!"]
>>> list(custom_replace_transform(list_a))
    ['sentencepiece encode as pieces', 'examples to try!']
```



## simple_space_split （简单空格分词器）

```python
torchtext.data.functional.simple_space_split(iterator)
```

用途：

​		一个将句子用空格分词的分词器

示例：

```python
>>> from torchtext.data.functional import simple_space_split
>>> list_a = ["Sentencepiece encode as pieces", "example to try!"]
>>> list(simple_space_split(list_a))
    [['Sentencepiece', 'encode', 'as', 'pieces'], ['example', 'to', 'try!']]
```



## numericalize_tokens_from_iterator (从迭代器中数字化令牌)

```python
torchtext.data.functional.numericalize_tokens_from_iterator(vocab, iterator, removed_tokens=None)
```

用途：

​		**Yield** a list of ids from an token iterator with a vocab. 从带有词汇的令牌迭代器中生成 id 列表。

参数如下：

- **vocab** – 将令牌转换为 id 的词汇表
- **iterator** – 用于产生令牌列表的迭代器
- **removed_tokens** – 从输出数据集中删除令牌（默认值：无）

示例：

```python
>>> from torchtext.data.functional import simple_space_split
>>> from torchtext.data.functional import numericalize_tokens_from_iterator
>>> vocab = {'Sentencepiece' : 0, 'encode' : 1, 'as' : 2, 'pieces' : 3}
>>> ids_iter = numericalize_tokens_from_iterator(vocab,
>>>                               simple_space_split(["Sentencepiece as pieces",
>>>                                                   "as pieces"]))
>>> for ids in ids_iter:
>>>     print([num for num in ids])
>>> [0, 2, 3]
>>> [2, 3]
```



## filter_wikipedia_xml

```python
torchtext.data.functional.filter_wikipedia_xml(text_iterator)
```

用途：

​		针对来自维基百科的数据集，从数据集中根据 https://github.com/facebookresearch/fastText/blob/master/wikifil.pl 过滤掉 维基百科xml 行

参数：

* **text_iterator** – An iterator type object that yields strings. Examples include string list, text io, generators etc.

示例：

```python
>>> from torchtext.data.functional import filter_wikipedia_xml
>>> from torchtext.datasets import EnWik9
>>> data_iter = EnWik9(split='train')
>>> filter_data_iter = filter_wikipedia_xml(data_iter)
>>> file_name = '.data/EnWik9/enwik9'
>>> filter_data_iter = filter_wikipedia_xml(open(file_name,'r'))
```



## to_map_style_dataset

```python
torchtext.data.functional.to_map_style_dataset(iter_data)
```

用途：

​		将可迭代样式数据集转换为映射样式（map-style）数据集。

参数：

* **iter_data** – An iterator type object. Examples include Iterable datasets, string list, text io, generators etc.

示例：

```python
>>> from torchtext.datasets import IMDB
>>> from torchtext.data import to_map_style_dataset
>>> train_iter = IMDB(split='train')
>>> train_dataset = to_map_style_dataset(train_iter)
>>> file_name = '.data/EnWik9/enwik9'
>>> data_iter = to_map_style_dataset(open(file_name,'r'))
```

