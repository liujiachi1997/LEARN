# 介绍python对象类型

对象是python中的“材料”，我们的代码就是对“材料”进行“操作事物”的过程。
常见的内置对象有：

|对象类型|字面量 / 构造示例|
|-|-|
|数字|1234|
|字符串|'spam'|
|列表|[1,2,3,4]|
|字典|{ 'food':'spam' , 'taste' : 'yum' }|
|元组|(1, 'spam', 4)|
|文件|open( 'egg.txt' )|
|集合|set( 'abc' ), { 'a', 'b', 'c' }|
|其他核心类型|类型、None、布尔型|
|程序单元类型|函数、模块、类|
|python实现相关类型|已编译代码、调用栈跟踪|

正如你输入

```
>>'spam'
```
Python 会自动判定为字符串类型，Python 是动态类型的（它会自动跟踪类型而不是要求你声明代码），并且它是强语言类型（你只能对一个对象进行适合该类型的所有操作）。

目的：积累一些内置类型的特性，并激发你的学习欲望

---
## 数字
* 数字及数字的运算（+ - * /）幂（**）
* 常用数学模块，如 math
    ```
    >>> import math
    >>> math.pi
    3.141592653589793 # 为了人性化的显示，请使用 print()
    >>> math.sqrt(85)
    9.219544457292887
    ```
* 随机数字生成器 / 随机选择器
    ```
    >>> import random
    >>> random.random()
    0.24562389525470185
    >>> random.choice([1,2,3,4])
    3
    ```
---
## 字符串
* 序列操作
    ```
    >>> S = 'spam'
    >>> len(S)  # 获取字符串长度
    4 

    # 切片操作
    >>> S[1:]
    'pam'
    >>> S
    'spam'
    >>> S[:3] 
    'spa'
    >>> S[0:3] 
    'spa'
    >>> S[:-1] 
    'spa'
    >>> S[:]   
    'spam'

    # 运算符操作
    >>> S      
    'spam'
    >>> S + 'xyz'
    'spamxyz'
    >>> S * 8     
    'spamspamspamspamspamspamspamspam'
    ```
* 不可变性

    当你创建 'spam' 字符串并想把第一个字母替换为 'z' 时，你不可能通过改变某一位置的值赋值给对象，你应该创建一个新的对象并赋值给原对象。
    ```
    >>> S
    'spam'
    >>> S[0] = 'z'
    Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
    TypeError: 'str' object does not support item assignment
    >>> S = 'z' + S[1:]
    >>> S
    'zpam'
    ```
    Python 中每一个对象都可以分为可变的或者不可变的，在核心类型中，数字、字符串和元组是不可变的；列表、字典、集合是可变的。
    严格来说，你可以在原位置改变文本，这需要你将它扩展成一个由独立字符组成的列表，通过列表操作后再重新拼接起来，另一种方法是用 bytearray 类型：
    ```
    >>> S = 'shrubbery'
    >>> L = list(S) 
    >>> L
    ['s', 'h', 'r', 'u', 'b', 'b', 'e', 'r', 'y']
    >>> L[1] = 'c'
    >>> ''.join(L)
    'scrubbery'

    >>> B = bytearray(b'spam')
    >>> B.extend(b'eggs')     
    >>> B                
    bytearray(b'spameggs')
    >>> B.decode()
    'spameggs'
    ```
* 特定类型的方法
    ```
    # 字符串类型独有的操作
    # find：传入子字符串的偏移量，没找到返回 -1
    # replace：对全局进行搜索替换
    >>> S = 'Spam'
    >>> S.find('am')
    2
    >>> S
    'Spam'
    >>> S.replace('pa', 'XYZ')
    'SXYZm'
    >>> S
    'Spam'

    >>> S.upper()
    'SPAM'
    >>> S.isalpha()  # 判定是不是字母
    True
    >>> S.isdigit()  # 判定是不是数字
    False

    >>> line = 'aaa,bbb,ccccc,dd\n' 
    >>> line       
    'aaa,bbb,ccccc,dd\n'
    >>> line.rstrip()   # 去掉行末的空格项 '\n'
    'aaa,bbb,ccccc,dd'
    >>> line.rstrip().split(',')  # split()方法
    ['aaa', 'bbb', 'ccccc', 'dd']
    >>> line = 'aaa,bbb,cccc,d|d'         
    >>> line.split(',')[-1].split('|')  
    ['d', 'd']

    # 在做数据报告时，需要用到的一些替换方法
    # 格式化高级替换
    >>> '%s eggs, and %s' % ('spam', 'SPAM!')
    'spam eggs, and SPAM!'
    >>> '{0} eggs, and {1}'.format('spam', 'SPAM!') 
    'spam eggs, and SPAM!'
    >>> '{} eggs, and {}'.format('spam', 'SPAM!')   
    'spam eggs, and SPAM!'

    # 格式化具有丰富的形式
    >>> '{:,.2f}'.format(296999.2567) 
    '296,999.26'
    ```
* 寻求帮助
    调用 dir() 函数，返回关于对象的所有属性（的列表）
    ```
    # 返回关于对象的所有属性
    >>> dir('spam') 
    ['__add__', '__class__', '__contains__', '__delattr__', '__dir__', 
    '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', 
    '__getitem__', '__getnewargs__', '__gt__', '__hash__', '__init__', 
    '__init_subclass__', '__iter__', '__le__', '__len__', '__lt__', 
    '__mod__', '__mul__', '__ne__', '__new__', '__reduce__', 
    '__reduce_ex__', '__repr__', '__rmod__', '__rmul__', '__setattr__', 
    '__sizeof__', '__str__', '__subclasshook__', 'capitalize', 
    'casefold', 'center', 'count', 'encode', 'endswith', 'expandtabs', 
    'find', 'format', 'format_map', 'index', 'isalnum', 'isalpha', 
    'isascii', 'isdecimal', 'isdigit', 'isidentifier', 'islower', 
    'isnumeric', 'isprintable', 'isspace', 'istitle', 'isupper', 'join', 
    'ljust', 'lower', 'lstrip', 'maketrans', 'partition', 'replace', 
    'rfind', 'rindex', 'rjust', 'rpartition', 'rsplit', 'rstrip', 
    'split', 'splitlines', 'startswith', 'strip', 'swapcase', 'title', 
    'translate', 'upper', 'zfill']
    
    # 查询关于方法的用法
    >>> help(S.replace) 
    Help on built-in function replace:

    replace(old, new, count=-1, /) method of builtins.str instance
        Return a copy with all occurrences of substring old replaced by new.

        count
            Maximum number of occurrences to replace.
            -1 (the default value) means replace all occurrences.

        If the optional argument count is given, only the first count occurrences are
        replaced.
    
    ```
* 字符串编程的其他方式
    ```
    >>> S = 'A\nB\tC'  # \n 是换行 \t 是tab   
    >>> len(S) 
    5
    ```
* Unicode 字符串
    ```
    >>> 'sp\xc4m'
    'spÄm'
    >>> u'sp\u00c4m'
    'spÄm'
    >>> 'spam'.encode('utf8') 
    b'spam'
    >>> 'spam'.encode('utf16') 
    b'\xff\xfes\x00p\x00a\x00m\x00'
    ```
* 模式匹配
    
    导入 re 模块，包含类似搜索、分割、替换等调用
    ```
    >>> match = re.match('[/:](.*)[/:](.*)[/:](.*)', '/usr/home/lumberjack')  
    >>> match.groups()
    ('usr', 'home', 'lumberjack')
    >>> re.split('[:/]', '/usr/home/lumberjack')
    ['', 'usr', 'home', 'lumberjack']
    ```
---
## 列表
* 序列操作
    
    同字符串上述
* 特定类型操作

    很多常见操作，比如末端加入单个元素：.append() / 多个元素.extend(),按位置删除 .pop() ，按值删除 .remove() 在任意位置插入 .insert() ，排序 .sort() 翻转列表 .reverse() 
* 边界检查
* 嵌套

    矩阵
    ```
    >>> M
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    >>> M[1]
    [4, 5, 6]
    >>> M[1][2] 
    6
    ```
* 推导
    ```
    >>> col2 = [row[1] for row in M]  # 获得矩阵的第二列
    >>> col2
    [2, 5, 8]
    >>> M
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # 矩阵并未改变
    >>> [row[1] + 1 for row in M]  # 复杂表示
    [3, 6, 9]
    >>> [row[1] for row in M if row[1] % 2 == 0] 
    [2, 8]
    >>> [M[i][i] for i in [0,1,2]] 
    [1, 5, 9]
    >>> [c * 2 for c in 'spam']
    ['ss', 'pp', 'aa', 'mm']
    #  range
    >>> list(range(4))
    [0, 1, 2, 3]
    >>> list(range(-6, 7, 2))  # 从-6到+6，间隔为2
    [-6, -4, -2, 0, 2, 4, 6]

    # 创建集合、字典、生成器
    >>> [ord(x) for x in 'spaam'] 
    [115, 112, 97, 97, 109]
    >>> {ord(x) for x in 'spaam'}  # 创建集合
    {112, 97, 115, 109}  
    >>> {x:ord(x) for x in 'spaam'}  # 创建字典
    {'s': 115, 'p': 112, 'a': 97, 'm': 109}
    >>> (ord(x) for x in 'spam')   # 创建生成器
    <generator object <genexpr> at 0x0000020DF73EF048>
    ```
---
## 字典
* 映射操作
    ```
    >>> D = {}
    >>> D['name'] = 'Tommy'
    >>> D['family'] = 'Peaky Blinders'
    >>> D['age'] = 40
    >>> D    # 创建了一个字典
    {'name': 'Tommy', 'family': 'Peaky Blinders', 'age': 40}
    >>> D['age'] += 1
    >>> D
    {'name': 'Tommy', 'family': 'Peaky Blinders', 'age': 41}
    #  用dict()创建
    >>> tom1 = dict(name = 'Tommy', family = 'Peaky Blinders', age = 40)
    >>> tom1
    {'name': 'Tommy', 'family': 'Peaky Blinders', 'age': 40}
    >>> tom1 = dict(zip(['name', 'family', 'age'], ['Tommy', 'Peaky Blinders',  40]))   # 用zip()配对映射
    >>> tom1
    {'name': 'Tommy', 'family': 'Peaky Blinders', 'age': 40}
    ```
* 重访嵌套
    
    字典中嵌套其他类型
    ```
    >>> rec = {'name':{'first':'Tom', 'last':'sherrby'}, 
    ... 'job':['buster', 'boss'],
    ... 'age':40}
    >>> rec                                             
    {'name': {'first': 'Tom', 'last': 'sherrby'}, 'job': ['buster', 'boss'], 'age': 40}
    >>> rec['name']
    {'first': 'Tom', 'last': 'sherrby'}
    >>> rec['name']['last']
    'sherrby'
    >>> rec['job'][-1]      
    'boss'
    ```
* 不存在的键：if 测试

    如果字典中不存在某个键，会返回错误
    ```
    >>> D = {'a' : 1, 'b' : 2, 'c' : 3} 
    >>> D['f']                                          
    Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
    KeyError: 'f'
    >>> value = D.get('a',0) 
    >>> value
    1
    >>> value = D.get('x',0) 
    >>> value
    0
    >>> value = D['x'] if 'x' in D else 0
    >>> value
    0
    ```
* 键的排序：for 循环

    字典是一对映射的集合，所以内部没有顺序，可以利用内置对象 sorted() 对其进行顺序打印：
    ```
    >>> D                        
    {'a': 1, 'c': 3, 'b': 2}
    >>> for key in sorted(D):            
    ...     print(key, '=>', D[key])
    ... 
    a => 1
    b => 2
    c => 3
    ```

* 迭代和优化

    字典满足迭代协议
---
## 元组
* 为什么要使用元组？

    因为相比于列表，元组虽然有更少的操作，但是他的 ‘不可变性’ 使他提供了一种完整性的约束。这对于大型程序是方便的
    ```
    >>> T = (1,2,3,4) 
    >>> T             
    (1, 2, 3, 4)
    >>> T + (5,6) 
    (1, 2, 3, 4, 5, 6)
    >>> T.count(4)  # '4'在元组中出现了1次
    1
    >>> T = 1,2,3,4,5  # 在创建元组时，圆括号可以被忽略
    >>> T             
    (1, 2, 3, 4, 5)
    >>> T.append(1)   # 不可以更改元组的内容
    Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
    AttributeError: 'tuple' object has no attribute 'append'
    ```
---
## 文件
* 创建、写入文件

    ```
    >>> f = open('data.txt','w')
    >>> f.write('Hello\n') #返回写入的字符串的字节数
    6
    >>> f.write('world\n') #依次调用write()方法,是以追加的方式写入内容,并不会覆盖前面的内容
    6
    >>> f.close()
    ```
* 读取文件

    ```
    >>> f = open('data.txt') #默认是以'r'的模式打开
    >>> text = f.read() #将文件以一个字符串的形式读出
    >>> text
    'Hello\nworld\n'

    >>> print(text)
    Hello
    world

    >>> text.split() #将字符串按照空白占位符(回车,tab,空格)进行分割,返回列表
    ['Hello','world']
    >>> f.seek(0) #将文件指针偏移值返回文件头
    >>> f.readline() #一行一行的读取
    'Hello\n'
    >>> f.readline()
    'world\n'
    >>> f.seek(0)
    >>> f.readlines() #将文件的内容一次性读取出来,保存到列表中,列表中的每一个元素是文件的一行
    ['Hello\n', 'world\n']
    ```
---
## 其他核心类型
* 集合

    集合更类似于一个无值的字典，里面是无序的键的集合
    ```
    >>> X = set('spam') #创建集合
    >>> X
    {'m', 'p', 'a', 's'}
    >>> Y = {'h','a','m'}
    >>> X,Y
    ({'m', 'p', 'a', 's'}, {'h', 'a', 'm'})
    >>> X & Y #集合的与运算
    {'a', 'm'}
    >>> X | Y #集合或运算
    {'p', 'h', 'a', 'm', 's'}
    >>> X - Y #集合减
    {'p', 's'}
    ```

* 检查对象的类型

    用 type() 来检查对象是否属于某个类型
    ```
    >>> L = [1,2,3]
    >>> if type(L) == type([]):   
    ...      print('yes')
    ... 
    yes
    >>> if type(L) == list:
    ...      print('yes')  
    ... 
    yes
    >>> if isinstance(L,list): #判断两个参数是否是同一个类型
    ...      print('yes')
    ... 
    yes
    ```