# 题库随机，语言随机

## 原创思路

### 供暖器II（别处做的，没找到原题）
题目：
> 冬季已经来临。你的任务是设计一个有固定加热半径的供暖器向所有房屋供暖。
>
> 在加热器的加热半径范围内的每个房屋都可以获得**一次**供暖。
>
> 现在，给出位于一条水平线上的供暖器`heaters`的位置和水平线上每个位置上的房屋需要的最少供暖次数`houses`，请你找出并返回可以覆盖所有房屋的最小加热半径。
>
> **说明**：所有供暖器都遵循你的半径标准，加热的半径也一样。
>
> **示例1**：
> ```
> 输入：houses = [1,1,1,1,1,1,1,2,1,1,1,1,1,2], heaters = [4,11,3,7]
> 输出：6
> 解释：位置7和13需要至少两次供暖，其余位置都只需一次供暖。我们需要将加热半径设为6，这样才能使用位置13的房屋得到两次供暖。
> ```
>
> **提示**：
> - 1 ≤ `heaters.length` ≤ `houses.length` ≤ 5 × 10<sup>4</sup>
> - 1 ≤ `houses[i], heaters[i]` ≤ 5 × 10<sup>4</sup>
> - `heaters`中没有重复元素

二分法：二分查找欲求半径（O(log n)）。判断当前半径是否符合要求可用差分数组加速区间+1操作，使判断操作O(n)。

- 时间复杂度：O(n log n)，其中n是数组houses的长度。
- 空间复杂度：O(n)，即为差分数组需要使用的空间。

贪心：每间房屋都至少要被周围最近的数个供暖器覆盖，因此可先排序`heaters`数组（O(n log n)），然后在遍历`houses`的同时用指针扫描`heaters`，从距离当前房屋最近的供暖器开始向两侧寻找`houses[i]`个供暖器。这是在两个排序数组中寻找第k大数问题，可二分解决（O(log n)）。

- 时间复杂度：O(n log n)，其中n是数组houses的长度。排序和遍历二分查找的时间复杂度都是O(n log n)。
- 空间复杂度：O(log n)。空间复杂度主要取决于排序所需要的空间。

### 067 最大的异或
二重二分法：先排序（O(n log n)），这样“寻找所有拥有指定前缀的数”的操作就加快到O(log n)。再按字典树的思路查找每个数对应的能使异或值最大的数，每个数要查找31次。

```python
def findMaximumXOR(self, nums: List[int]) -> int:
    nums.sort()
    ans = 0
    for i in range(len(nums)):
        y = 0
        lo = i
        hi = len(nums)
        # 最高位的二进制位编号为30
        for b in range(30, -1, -1):
            mid = bisect_left(nums, y | 1 << b, lo=lo, hi=hi)
            # 如果不存在当前位与nums[i]相反的数，那么只能相同了
            if mid == hi or mid != lo and nums[i] & 1 << b:
                hi = mid
            else:
                lo = mid
                y |= 1 << b
            if lo == hi:
                break
            # 当范围中所有数都相同时，找到了最大值
            if nums[lo] == nums[hi - 1]:
                ans = max(ans, nums[i] ^ nums[lo])
                break
    return ans
```

- 时间复杂度：O(n log n log C)，其中n是数组nums的长度，C是数组中的元素范围。
- 空间复杂度：O(log n)，其中n是数组nums的长度。空间复杂度主要取决于排序所需要的空间。

### 081 允许重复选择元素的组合
没什么特别的，只是想让你们看看二重for循环list comprehension。

```python
def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
    x = candidates[-1]
    if len(candidates) == 1:
        return [] if target % x else [target // x * candidates]
    if target == 0:
        return [[]]
    return [s + [x] * i for i in range(target // x + 1) for s in self.combinationSum(candidates[:-1], target - x * i)]
```

### 085 生成匹配的括号
棋盘格边上的递归：

```python
def generateParenthesis(self, n: int) -> List[str]:
    def p(l, r):
        if r == 0:
            return ["(" * l]
        if l < r:
            return []
        return [x + ")" for x in p(l, r - 1)] + [x + "(" for x in p(l - 1, r)]
    return p(n, n)
```

- 时间复杂度：O(4<sup>n</sup>/n<sup>1/2</sup>)，该分析与官方题解类似。
- 空间复杂度：O(4<sup>n</sup>/n<sup>1/2</sup>)，此方法除答案数组外，中间过程中会存储与答案数组同样数量级的临时数组，是我们所需要的空间复杂度。

### 101 分割等和子集
Pythonic动态规划变体：不用布尔数组，而是用集合。集合表示截止当前元素，通过选择可达的和。

```python
def canPartition(self, nums: List[int]) -> bool:
    target = sum(nums)
    if target % 2:
        return False
    target //= 2
    m = {0}
    for x in nums:
        m |= {s + x for s in m if s + x <= target}
        if target in m:
            return True
    return False
```

- 时间复杂度：O(n × target)，其中n是数组的长度，target是整个数组的元素和的一半。需要计算出所有的状态，每个状态在进行转移时的时间复杂度为O(1)。
- 空间复杂度：O(target)，其中target是整个数组的元素和的一半。空间复杂度取决于`m`集合，其中最多有target + 1个元素。

### 102 加减的目标值
Pythonic动态规划变体：不用数组，而是用字典。键值对表示截止当前元素，加减能够达成指定值的不同表达式数目。

```python
def findTargetSumWays(self, nums: List[int], target: int) -> int:
    m = {0: 1}
    for x in nums:
        m2 = defaultdict(int)
        for s in m:
            m2[s + x] += m[s]
            m2[s - x] += m[s]
        m = m2
    return m[target]
```

- 时间复杂度：O(n × sum)，其中n是数组`nums`的长度，sum是数组`nums`的元素和。字典中最多存在sum × 2 + 1个键，需要计算每个键的值。
- 空间复杂度：O(sum)，其中sum是数组`nums`的元素和。空间复杂度取决于字典中键值对的数目。

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
- 048 序列化与反序列化二叉树
  ```javascript
  var serialize = JSON.stringify, deserialize = JSON.parse
  ```
- 060 出现频率最高的k个数字
  ```python
  return [x[0] for x in heapq.nlargest(k, collections.Counter(nums).items(), key=lambda x: x[1])]
  ```
- 066 单词之和
  ```python
  def __init__(self):
      self.m = {}

  def insert(self, key: str, val: int) -> None:
      self.m[key] = val

  def sum(self, prefix: str) -> int:
      return sum(v for k, v in self.m.items() if k[:len(prefix)] == prefix)
  ```
- 072 求平方根
  ```scheme
  (inexact->exact (floor (sqrt x)))
  ```
- 073 狒狒吃香蕉 ← 官方题解
  ```python
  return bisect_left(range(max(piles)), -h, 1, key=lambda k: -sum((pile + k - 1) // k for pile in piles))
  ```
- 076 数组中的第k大的数字
  ```python
  return heapq.nlargest(k, nums)[-1]
  ```
- 079 所有子集
  ```python
  return [[x for j, x in enumerate(nums) if i & 1 << j] for i in range(1 << len(nums))]
  ```
- 080 含有k个元素的组合
  ```python
  return list(itertools.combinations(range(1, n + 1), k))
  ```
- 083 没有重复元素集合的全排列
  ```python
  return list(itertools.permutations(nums))
  ```
- 086 分割回文子字符串
  ```python
  return [[s[:i]] + t for i in range(1, len(s) + 1) if s[:i] == s[i-1::-1] for t in self.partition(s[i:])] if s else [[]]
  ```
- 088 爬楼梯的最少成本
  ```python
  return min(reduce(lambda s, x: (s[1], min(s) + x), cost, (0, 0)))
  ```
- 089 房屋偷盗
  ```c
  for (int a = 0, b = 0, c; ; a = b, b = c) if (c = fmax(a + nums[--numsSize], b), !numsSize) return c;
  ```
- 092 翻转字符
  ```c
  for (int a = 0, b = 0; ; ) if (*s - 48 ? a = fmin(a, b), b++: a++, !*s++) return a;
  ```
- 098 路径的数目
  ```ruby
  (3..(m + n)).to_a.combination(m - 1).size
  ```
- 100 三角形中最小路径之和
  ```ruby
  triangle.reverse.reduce { |s, x| s.each_cons(2).map(&:min).zip(x).map(&:sum) }.first
  ```
- 119 最长连续序列
  ```python
  nums = set(nums); return max((len(list(takewhile(lambda x: x in nums, count(i)))) for i in nums if i - 1 not in nums), default=0)
  ```
- 2425 所有数对的异或和
  ```ruby
  (nums2.size.odd? ? nums1.reduce(:^) : 0) ^ (nums1.size.odd? ? nums2.reduce(:^) : 0)
  ```
- 2429 最小XOR
  ```ruby
  ("%032b" % num1).tap { |s| num2.to_s(2).count("1").times { s[s.rindex("0")] = "x" if !s.sub!("1", "x") } }.tr("1x", "01").to_i(2)
  ```

## 极致的优雅
- 065 最短的单词编码 ← 官方题解
  ```python
  def minimumLengthEncoding(self, words: List[str]) -> int:
      # 删除重复元素
      words = list(set(words))
      # 字典项不存在时自动再创建一个defaultdict
      Trie = lambda: collections.defaultdict(Trie)
      trie = Trie()
      # reduce(dict.__getitem__, S, trie)表示trie[S[0]][S[1]][S[2]][...][S[len(S) - 1]]
      nodes = [reduce(dict.__getitem__, word[::-1], trie) for word in words]
      # 没有相邻项的节点即是答案
      return sum(len(word) + 1 for i, word in enumerate(words) if len(nodes[i]) == 0)
  ```

## 很难绷得住
C和Go在做算法题时是难兄难弟，标准库里啥都没有，最长题解必有它俩……

- 047 二叉树剪枝

  递归，优雅！  
  你是一个一个一个带有GC的语言啊啊啊啊啊  
  让我们看看C语言题解：  
  C语言题解直接内存泄漏！
