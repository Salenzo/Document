# 剑指 Offer II

## 原创思路

## 一行代码赖皮过法

- 001 整数除法
  ```python
  return min(int(__import__('operator').itruediv(a, b)), 2 ** 31 - 1)
  ```
- 002 二进制加法
  ```python
  return bin(int(a, 2) + int(b, 2))[2:]
  ```
- 018 有效的回文
  ```python
  return (lambda s: s[::-1] == s)(__import__('re').sub(r'[\W_]', '', s).casefold())
  ```
- 032 有效的变位词
  ```ruby
  s != t && s.each_char.sort == t.each_char.sort
  ```
- 033 变位词组
  ```python
  h = collections.defaultdict(list); [h[''.join(sorted(w))].append(w) for w in strs]; return list(h.values())
  ```
- 034 外星语言是否排序
  ```ruby
  words.map { |x| x.tr(order, "a-z") }.each_cons(2).all? { |a, b| a <= b }
  ```
- 035 最小时间差
  ```ruby
  time_points.map { |t| t[0..1].to_i * 60 + t[3..4].to_i }.flat_map { |t| [t, t + 1440] }.sort.each_cons(2).map { |a, b| b - a }.min
  ```
- 060 出现频率最高的k个数字
  ```python
  return [x[0] for x in heapq.nlargest(k, collections.Counter(nums).items(), key=lambda x: x[1])]
  ```
