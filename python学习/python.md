# 基础

## 调用解释器

1. 首先把python解释器添加到计算机环境变量中

2. cmd中输入python
3. windows中输入control+z推出解释器/ 或者输入quit()

## 交互模式

在终端（tty）输入并执行指令时，解释器在 *交互模式（interactive mode）* 中运行。在这种模式中，会显示 *主提示符*，提示输入下一条指令，主提示符通常用三个大于号（`>>>`）表示；输入连续行时，显示 *次要提示符*，默认是三个点（`...`）。进入解释器时，首先显示欢迎信息、版本信息、版权声明，然后才是提示符：

```
>>> the_world_is_flat = True
>>> if the_world_is_flat:
...     print("Be careful not to fall off!")
...
Be careful not to fall off!
```

## 解释器的运行环境

默认情况下，Python 源码文件的编码是 UTF-8。这种编码支持世界上大多数语言的字符，可以用于字符串字面值、变量、函数名及注释 —— 尽管标准库只用常规的 ASCII 字符作为变量名或函数名，可移植代码都应遵守此约定。要正确显示这些字符，编辑器必须能识别 UTF-8 编码，而且必须使用支持文件中所有字符的字体。

如果不使用默认编码，则要声明文件的编码，文件的 *第一* 行要写成特殊注释。句法如下：

```
# -*- coding: encoding -*-
```

其中，*encoding* 可以是 Python 支持的任意一种 [`codecs`](https://docs.python.org/zh-cn/3/library/codecs.html#module-codecs)。

比如，声明使用 Windows-1252 编码，源码文件要写成：

```
# -*- coding: cp1252 -*-
```

## 速览

- Python 注释以 `#` 开头，直到该物理行结束。注释可以在行开头，或空白符与代码之后，但不能在字符串里面。

**python计算**

- 除法运算（`/`）返回浮点数。用 `//` 运算符执行 [floor division](https://docs.python.org/zh-cn/3/glossary.html#term-floor-division) 的结果是整数（忽略小数）；计算余数用 `%`

  ```
  >>> 17 / 3  # classic division returns a float
  5.666666666666667
  >>>
  >>> 17 // 3  # floor division discards the fractional part
  5
  >>> 17 % 3  # the % operator returns the remainder of the division
  2
  >>> 5 * 3 + 2  # floored quotient * divisor + remainder
  17
  ```

- 用 `**` 运算符计算乘方

  ```
  >>> 5 ** 2  # 5 squared
  25
  >>> 2 ** 7  # 2 to the power of 7
  128
  ```

- **如果变量未定义（即，未赋值），使用该变量会提示错误：**

  ```
  >>> n  # try to access an undefined variable
  Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
  NameError: name 'n' is not defined
  ```

- Python 全面支持浮点数；混合类型运算数的运算会把整数转换为浮点数：

  ```
  >>> 4 * 3.75 - 1
  14.0
  ```

- **交互模式下，上次输出的表达式会赋给变量 `_`**。把 Python 当作计算器时，用该变量实现下一步计算更简单，例如：

  - 最好把该变量当作只读类型。不要为它显式赋值，否则会创建一个同名独立局部变量，该变量会用它的魔法行为屏蔽内置变量。

  ```
  >>> tax = 12.5 / 100
  >>> price = 100.50
  >>> price * tax
  12.5625
  >>> price + _
  113.0625
  >>> round(_, 2)
  113.06
  ```

**字符串**

- 除了数字，Python 还可以操作字符串。字符串有多种表现形式，用单引号（`'……'`）或双引号（`"……"`）标注的结果相同。

- 反斜杠 `\` 用于转义：

- 交互式解释器会为输出的字符串加注引号，特殊字符使用反斜杠转义。虽然，有时输出的字符串看起来与输入的字符串不一样（外加的引号可能会改变），但两个字符串是相同的。如果字符串中有单引号而没有双引号，该字符串外将加注双引号，反之，则加注单引号。[`print()`](https://docs.python.org/zh-cn/3/library/functions.html#print) 函数输出的内容更简洁易读，它会省略两边的引号，并输出转义后的特殊字符：

  ```
  >>> '"Isn\'t," they said.'
  '"Isn\'t," they said.'
  >>> print('"Isn\'t," they said.')
  "Isn't," they said.
  >>> s = 'First line.\nSecond line.'  # \n means newline
  >>> s  # without print(), \n is included in the output
  'First line.\nSecond line.'
  >>> print(s)  # with print(), \n produces a new line
  First line.
  Second line.
  ```




## 1、字符串前加 u

例：u"我是含有中文字符组成的字符串。"

作用：

后面字符串以 Unicode 格式 进行编码，一般用在中文字符串前面，防止因为源码储存格式问题，导致再次使用时出现乱码。

## 2、字符串前加 r

例：r"\n\n\n\n”　　# 表示一个普通生字符串 \n\n\n\n，而不表示换行了。

作用：

去掉反斜杠的转移机制。

（特殊字符：即那些，反斜杠加上对应字母，表示对应的特殊含义的，比如最常见的”\n”表示换行，”\t”表示Tab等。 ）

应用：

常用于正则表达式，对应着re模块。

## 3、字符串前加 b

例: response = b'<h1>Hello World!</h1>'   # b' ' 表示这是一个 bytes 对象

作用：

b" "前缀表示：后面字符串是bytes 类型。

用处：

网络编程中，服务器和浏览器只认bytes 类型数据。

如：*send 函数的参数和 recv 函数的返回值都是 bytes 类型*

附：

在 Python3 中，bytes 和 str 的互相转换方式是
str.encode('utf-8')
bytes.decode('utf-8')

## 4、字符串前加 f

import time
t0 = time.time()
time.sleep(1)
name = 'processing'

*# 以 **f**开头表示在字符串内支持大括号内的python 表达式*
print(f'{name} done in {time.time() - t0:.2f} s') 

**输出：**
processing done in 1.00 s









# yield关键字

您可能听说过，带有 yield 的函数在 Python 中被称之为 generator（生成器），何谓 generator ？

我们先抛开 generator，以一个常见的编程题目来展示 yield 的概念。

## 一、说明

return一直中，每中语言中其没没有很大差别，就不多说了。（shell语言return的是退出状态，可能差别是比较大的，感兴趣可参见“[Linux Shell函数定义与调用](https://www.cnblogs.com/lsdb/p/10148177.html)”）

最早看到yield应该是哪们语言用来调整什么线程优先级的，记不清了，不过那里的yield和python中的yield应该功能有区别。

python中最早看到yield应该是使用scrapy框架写爬虫的时候，之前也有去看yiled的用法，总记不太住。今天又去看了一下，基本上来就是讲些斐波那契数列的烦的要死，自己写段程序研究了一下，这里记一下。

 

## 二、return和yield的异同

共同点：return和yield都用来返回值；在一次性地返回所有值场景中return和yield的作用是一样的。

不同点：如果要返回的数据是通过for等循环生成的迭代器类型数据（如列表、元组），return只能在循环外部一次性地返回，yeild则可以在循环内部逐个元素返回。下边我们举例说明这个不同点。

从上边两个小节可以看到，虽然return和yield两者执行的顺序有区别，但整个要做的事情是一样的，所以使用yield并不会比return快，甚至我们可以猜测由于yield总发生上下文切换在速度上还会慢一些，所以速度不是yield的意义。

他们的主要区别是yiled要迭代到哪个元素那个元素才即时地生成，而return要用一个中间变量result_list保存返回值，当result_list的长度很长且每个组成元素内容很大时将会耗费比较大的内存，此时yield相对return才有优势。

# with关键字



Python 中的 **with** 语句用于异常处理，封装了 **try…except…finally** 编码范式，提高了易用性。

**with** 语句使代码更清晰、更具可读性， 它简化了文件流等公共资源的管理。

在处理文件对象时使用 with 关键字是一种很好的做法。

我们可以看下以下几种代码实例：

不使用 **with**，也不使用 **try…except…finally**

with 语句实现原理建立在上下文管理器之上。

上下文管理器是一个实现 **__enter__** 和 **__exit__** 方法的类。

使用 with 语句确保在嵌套块的末尾调用 __exit__ 方法。

这个概念类似于 try...finally 块的使用。

