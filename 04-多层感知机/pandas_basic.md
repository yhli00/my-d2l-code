# pandas基础操作
## 创建DataFrame、Series
`pd.Series`
```python
>>> a = {'a':1, 'b':2, 'c':3}
>>> b = pd.Series(a, index=['a', 'b', 'c'])
>>> b
a    1
b    2
c    3
dtype: int64
>>> b = pd.Series(a)
>>> b
a    1
b    2
c    3
dtype: int64
```
```python
>>> a = {'a':1, 'b':2, 'c':3}
>>> b = pd.Series(a, index=['x', 'y', 'z'])
>>> b
x   NaN
y   NaN
z   NaN
dtype: float64
```
```python
>>> a = pd.Series(['a', 'b', 'c'])
>>> a
0    a
1    b
2    c
dtype: object
```
`pd.DataFrame`
```python
# 数据类型
>>> df = pd.DataFrame({'float': [1.0],
    'int': [1],
    'datetime': [pd.Timestamp('20180310')],
    'string': ['foo']}
)
>>> df.dtypes
float              float64
int                  int64
datetime    datetime64[ns]
string              object
dtype: object
```
```python
# 直接构建
>>> d = {'col1': [1, 2], 'col2': [3, 4]}
>>> df = pd.DataFrame(data=d)
>>> df
   col1  col2
0     1     3
1     2     4
>>> d = {'col1': [1, 2], 'col2': [3]}
>>> df = pd.DataFrame(data=d)

>>> d = {'col1': [1, 2], 'col2': [3]}
>>> df = pd.DataFrame(data=d)  # 报错
```
```python
# 从numpy构建
>>> df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['a', 'b', 'c'])
>>> df
   a  b  c
0  1  2  3
1  4  5  6
2  7  8  9
>>> df.dtypes
a    int32
b    int32
c    int32
dtype: object
```
## 读写文件
`pd.to_csv`: 把DataFrame写入以逗号分隔的csv文件，index=False表示不写入行号

`pd.read_csv`: 把以逗号分隔的csv文件读取成DataFrame
```python
>>> df = pd.DataFrame({'name': ['Raphael',          'Donatello'],
...                    'mask': ['red', 'purple'],
...                    'weapon': ['sai', 'bo staff']})
>>> df
        name    mask    weapon
0    Raphael     red       sai
1  Donatello  purple  bo staff
>>> df.to_csv('../test.csv', index=False)

>>> pd.read_csv('../test.csv')
        name    mask    weapon
0    Raphael     red       sai
1  Donatello  purple  bo staff
# csv中的实际内容：
# name,mask,weapon
# Raphael,red,sai
# Donatello,purple,bo staff
```
## 基本操作
`df.iloc[]`索引
```python
>>> mydict = [{'a': 1, 'b': 2, 'c': 3, 'd': 4},
...           {'a': 100, 'b': 200, 'c': 300, 'd': 400},
...           {'a': 1000, 'b': 2000, 'c': 3000, 'd': 4000 }]  # 按行创建
>>> df = pd.DataFrame(mydict)
>>> df
      a     b     c     d
0     1     2     3     4
1   100   200   300   400
2  1000  2000  3000  4000

>>> df.iloc[0]
a    1
b    2
c    3
d    4
Name: 0, dtype: int64
>>> type(df.iloc[0])
<class 'pandas.core.series.Series'>
>>> df.iloc[[0]]
   a  b  c  d
0  1  2  3  4
>>> type(df.iloc[[0]])
<class 'pandas.core.frame.DataFrame'>

>>> df.iloc[[0, 2]]
      a     b     c     d
0     1     2     3     4
2  1000  2000  3000  4000

>>> df.iloc[0:2, 1:2]
     b
0    2
1  200
```
`pd.DataFrame`可以直接按列索引，不能按行索引
```python
>>> df = pd.DataFrame([[1, 2, 3]] * 3, columns=['A', 'B', 'C'])
>>> df
   A  B  C
0  1  2  3
1  1  2  3
2  1  2  3
>>> df['A']
0    1
1    1
2    1
Name: A, dtype: int64
>>> df[['A', 'C']]
   A  C
0  1  3
1  1  3
2  1  3
>>> df[0]  # 报错
```
```
`df.shape`和`df.values`
```python
>>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
>>> df
   a  b  c
0  1  4  7
1  2  5  8
2  3  6  9
>>> df.shape
(3, 3)

>>> df.values
array([[1, 4, 7],
       [2, 5, 8],
       [3, 6, 9]], dtype=int64)
>>> type(df.values)
<class 'numpy.ndarray'>
```
`df.apply`
```python
>>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
>>> df
   A  B  C
0  1  4  7
1  2  5  8
2  3  6  9
>>> df.apply(lambda x: (x - x.mean()) / x.std())  # 对每一列的数据归一化
     A    B    C
0 -1.0 -1.0 -1.0
1  0.0  0.0  0.0
2  1.0  1.0  1.0
>>> df.apply(lambda x: (x - x.mean()) / x.std(), axis=1)  # 对每一行的数据归一化
     A    B    C
0 -1.0  0.0  1.0
1 -1.0  0.0  1.0
2 -1.0  0.0  1.0
```
`df.fillna`
```python
>>> df = pd.DataFrame([[1, 2, np.nan], [3, np.nan, 4], [np.nan, 5, 6]], columns=list('ABC'))
>>> df
     A    B    C
0  1.0  2.0  NaN
1  3.0  NaN  4.0
2  NaN  5.0  6.0
>>> df.fillna(0)
     A    B    C
0  1.0  2.0  0.0
1  3.0  0.0  4.0
2  0.0  5.0  6.0
```
`pd.get_dummies`表示将离散值表示成one-hot形式，参数dummy_na表示将nan视为有效的特征
```python
>>> ser = pd.Series(list('abcd'))
>>> ser
0    a
1    b
2    c
3    d
dtype: object
>>> pd.get_dummies(ser)
   a  b  c  d
0  1  0  0  0
1  0  1  0  0
2  0  0  1  0
3  0  0  0  1


>>> df = pd.DataFrame([[1, 2, 'a', 'b'], [2, 3, np.nan, 'c']], columns=list('ABCD'))
>>> df
   A  B    C  D
0  1  2    a  b
1  2  3  NaN  c
>>> pd.get_dummies(df)
   A  B  C_a  D_b  D_c
0  1  2    1    1    0
1  2  3    0    0    1
>>> pd.get_dummies(df, dummy_na=True)  # 将nan视为有效特征
   A  B  C_a  C_nan  D_b  D_c  D_nan
0  1  2    1      0    1    0      0
1  2  3    0      1    0    1      0
```