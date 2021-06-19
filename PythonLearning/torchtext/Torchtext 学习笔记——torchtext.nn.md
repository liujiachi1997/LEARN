# Torchtext 学习笔记——torchtext.nn

**torchtext** 包由 ***数据处理实用程序*** 和流行的 ***自然语言数据集*** 组成。

## Torchtext.NN

### MultiheadAttentionContainer

这是一个多头注意力容器

```python
class torchtext.nn.MultiheadAttentionContainer(nhead, in_proj_container, attention_layer, out_proj, batch_first=False):
    __init__(nhead, in_proj_container, attention_layer, out_proj, batch_first=False)
```

参数如下：

* **nhead** – 多头注意力模型中的头数
* **in_proj_container** – 多头投影线性层（又名 nn.Linear）的容器
* **attention_layer** – 自定义注意力层
* **out_proj** – 多头输出投影层（又名 nn.Linear）
* **batch_first** – 如果为 True，则输入和输出张量以 (..., N, L, E) 形式提供。 默认值：假

例子：

```python
>>> import torch
>>> from torchtext.nn import MultiheadAttentionContainer, InProjContainer, ScaledDotProduct
>>> embed_dim, num_heads, bsz = 10, 5, 64
>>> in_proj_container = InProjContainer(torch.nn.Linear(embed_dim, embed_dim),
                                        torch.nn.Linear(embed_dim, embed_dim),
                                        torch.nn.Linear(embed_dim, embed_dim))
>>> MHA = MultiheadAttentionContainer(num_heads,
                                      in_proj_container,
                                      ScaledDotProduct(),
                                      torch.nn.Linear(embed_dim, embed_dim))
>>> query = torch.rand((21, bsz, embed_dim))
>>> key = value = torch.rand((16, bsz, embed_dim))
>>> attn_output, attn_weights = MHA(query, key, value)
>>> print(attn_output.shape)
>>> torch.Size([21, 64, 10])
```

```python
forward(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, bias_k: Optional[torch.Tensor] = None, bias_v: Optional[torch.Tensor] = None) → Tuple[torch.Tensor, torch.Tensor]
```

关于 MHA 的 forward() 参数如下：

* **query** (*Tensor*) – 注意函数的查询 **query**
* **key** (*Tensor*) – 注意函数的键值 **keys**
* **value** (*Tensor*) – 注意函数的值 **value**
* **attn_mask** (*BoolTensor**,* *optional*) – 防止注意某些位置的 3D mask
* **bias_k** (*Tensor**,* *optional*) – one more key and value sequence to be added to keys at sequence dim (dim=-3). Those are used for incremental decoding. Users should provide bias_v.
* **bias_v** (*Tensor**,* *optional*) – one more key and value sequence to be added to values at sequence dim (dim=-3). Those are used for incremental decoding. Users should also provide bias_k.

Shape:

> - Inputs:
>
>   > - query: (...,L,N,E)(...,L,N,E)
>   > - key: (...,S,N,E)(...,S,N,E)
>   > - value: (...,S,N,E)(...,S,N,E)
>   > - attn_mask, bias_k and bias_v: same with the shape of the corresponding args in attention layer.
>
> - Outputs:
>
>   > - attn_output: (...,L,N,E)(...,L,N,E)
>   > - attn_output_weights: (N∗H,L,S)

注意：查询/键/值输入具有三个以上的维度（用于广播目的）是可选的。 MultiheadAttentionContainer 模块将在最后三个维度上运行。

其中 L 是目标长度，S 是序列长度，H 是注意力头的数量，N 是批量大小，E 是嵌入维度。



### InProjContainer

```python
class torchtext.nn.InProjContainer(query_proj, key_proj, value_proj):
    __init__(query_proj, key_proj, value_proj)
```

这是在MultiheadAttention 中投影 Q/K/V 的容器，此模块发生在将投影的 Q/K/V 重塑为多个头之前。请参阅Attention Is All You Need 论文的图 2 中多头注意力的线性层（底部）。 还请参阅 torchtext.nn.MultiheadAttentionContainer 中的用法示例（上个示例）。

参数如下：

- **query_proj** – 用于 Q（查询）的项目层。 典型的投影层是 torch.nn.Linear。
- **key_proj** – 用于 K（键值）的项目层。 典型的投影层是 torch.nn.Linear。
- **value_proj** – 用于 V（值）的项目层。 典型的投影层是 torch.nn.Linear。

```python
forward(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) → Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```

使用 in-proj 层投影输入序列。 query/key/value 分别简单地传递给 query/key/value_proj 的前向函数。

参数如下：

- **query** (*Tensor*) – 要投影的查询。
- **key** (*Tensor*) – 要投影的键值。
- **value** (*Tensor*) – 要投影的值。

示例：

```python
>>> import torch
>>> from torchtext.nn import InProjContainer
>>> embed_dim, bsz = 10, 64
>>> in_proj_container = InProjContainer(torch.nn.Linear(embed_dim, embed_dim),
                                        torch.nn.Linear(embed_dim, embed_dim),
                                        torch.nn.Linear(embed_dim, embed_dim))
>>> q = torch.rand((5, bsz, embed_dim))
>>> k = v = torch.rand((6, bsz, embed_dim))
>>> q, k, v = in_proj_container(q, k, v)
```



### ScaledDotProduct (缩放点积)

```python
class torchtext.nn.ScaledDotProduct(dropout=0.0, batch_first=False):
    __init__(dropout=0.0, batch_first=False)
```

处理投影查询和键值对以应用 **缩放的点积** 注意力。

参数如下：

- **dropout** ([*float*](https://docs.python.org/3/library/functions.html#float)) – 降低注意力权重的概率。
- **batch_first** – 如果为`True`，则输入和输出张量以（batch、seq、feature）形式提供。 默认值：`False`

示例：

```python
>>> import torch, torchtext
>>> SDP = torchtext.nn.ScaledDotProduct(dropout=0.1)
>>> q = torch.randn(21, 256, 3)
>>> k = v = torch.randn(21, 256, 3)
>>> attn_output, attn_weights = SDP(q, k, v)
>>> print(attn_output.shape, attn_weights.shape)
torch.Size([21, 256, 3]) torch.Size([256, 21, 21])
```

```python
forward(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, bias_k: Optional[torch.Tensor] = None, bias_v: Optional[torch.Tensor] = None) → Tuple[torch.Tensor, torch.Tensor]
```

使用带有投影键值对的缩放点积来更新投影查询。

参数如下：

- **query** (*Tensor*) – 投影 query
- **key** (*Tensor*) – 投影 key
- **value** (*Tensor*) – 投影 value
- **attn_mask** (*BoolTensor**,* *optional*) – 防止注意某些位置的 3D mask
- **attn_mask** – 防止注意某些位置的 3D mask
- **bias_k** (*Tensor**,* *optional*) – one more key and value sequence to be added to keys at sequence dim (dim=-3). Those are used for incremental decoding. Users should provide `bias_v`.
- **bias_v** (*Tensor**,* *optional*) – one more key and value sequence to be added to values at sequence dim (dim=-3). Those are used for incremental decoding. Users should also provide `bias_k`.

Shape:

- query: (...,L,N∗H,E/H)(...,L,N∗H,E/H)

- key: (...,S,N∗H,E/H)(...,S,N∗H,E/H)

- value: (...,S,N∗H,E/H)(...,S,N∗H,E/H)

- - attn_mask: (N∗H,L,S)(N∗H,L,S), positions with `True` are not allowed to attend

    while `False` values will be unchanged.

- bias_k and bias_v:bias: (1,N∗H,E/H)(1,N∗H,E/H)

- Output: (...,L,N∗H,E/H)(...,L,N∗H,E/H), (N∗H,L,S)(N∗H,L,S)

- Note: 有超过三个维度的查询/键/值输入是可选的（用于广播目的）。

  ScaledDotProduct 模块将在最后三个维度上运行。

其中 L 是目标长度，S 是源长度，H 是注意力头的数量，N 是批量大小，E 是嵌入维度。