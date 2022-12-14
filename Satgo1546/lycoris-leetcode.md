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

### 112 最长递增路径
迫真记忆化广度优先搜索，非常慢，强烈不推荐。

```python
class Solution:
    def longestIncreasingPath(self, matrix: List[List[int]]) -> int:
        q = deque((i, j) for i in range(len(matrix)) for j in range(len(matrix[0])))
        ans = [[1] * len(matrix[0]) for _ in range(len(matrix))]
        while q:
            i, j = q.popleft()
            flag = False
            for i1, j1 in ((i+1, j), (i, j+1), (i-1, j), (i, j-1)):
                if 0 <= i1 < len(matrix) and 0 <= j1 < len(matrix[0]):
                    if matrix[i1][j1] < matrix[i][j] and ans[i1][j1] + 1 > ans[i][j]:
                        ans[i][j] = ans[i1][j1] + 1
                        flag = True
            if flag:
                for i1, j1 in ((i+1, j), (i, j+1), (i-1, j), (i, j-1)):
                    if 0 <= i1 < len(matrix) and 0 <= j1 < len(matrix[0]):
                        q.append((i1, j1))
        return max(max(row) for row in ans)
```

- 执行用时：7228 ms，在所有Python 3提交中击败了5.07%的用户。
- 内存消耗：41.7 MB，在所有Python 3提交中击败了5.07%的用户。

### 169 多数元素
基于快速排序的选择算法：寻找第⌊n/2⌋大元素，选择选择算法。

- 时间复杂度：O(n)，选择算法的时间代价的期望为线性。
- 空间复杂度：O(log n)，递归使用栈空间的空间代价的期望为O(log n)。

一句话解释清楚Boyer–Moore多数投票算法：遍历数组的时候，让不相同的数字两两抵消，那么最后剩下的肯定就是众数了。

### 878 第N个神奇数字
数学：[1, n]内有⌊n/a⌋ + ⌊n/b⌋ − ⌊n/lcm(a, b)⌋个a或b的倍数。若将a和b的公倍数当作两个数，则可以通过公式直接计算出第n个a或b的倍数。

```python
def nthMagicalNumber(self, n: int, a: int, b: int) -> int:
    d = gcd(a, b)
    n += n // (a // d + b // d - 1)
    return (min(a * ceil(n * b / (a + b)), b * ceil(n * a / (a + b)))) % 1000000007
```

- 时间复杂度：O(1)，中间结果可能超过32位整数，但不会超过64位整数。
- 空间复杂度：O(1)，仅使用常量空间开销。

## 一行代码赖皮过法

- 04 二维数组中的查找
  ```ruby
  matrix.flatten.include?(target)
  ```
- 05 替换空格
  ```php
  return str_replace(" ", "%20", $s);
  ```
- 09 用两个栈实现队列
  ```ruby
  class CQueue < Array
    alias append_tail push
    def delete_head = shift || -1
  end
  ```
- 11 旋转数组的最小数字
  ```ruby
  numbers.min
  ```
- 14-I 剪绳子
  ```ruby
  (2..n).map { |m| (n/m)**(m-n%m) * (n/m+1)**(n%m) }.max
  ```
- 14-II 剪绳子
  ```ruby
  (2..n).map { |m| (n/m)**(m-n%m) * (n/m+1)**(n%m) }.max % 1000000007
  ```
- 15 二进制中1的个数
  ```python
  return n.bit_count()
  ```
- 16 数值的整数次方
  ```scheme
  (define my-pow expt)
  ```
- 17 打印从1到最大的n位数
  ```rust
  (1..10i32.pow(n as u32)).collect()
  ```
- 19 正则表达式匹配
  ```ruby
  /\A#{p}\z/ === s
  ```
- 20 表示数值的字符串
  ```ruby
  /\A *[+-]?(\d*\.?\d+|\d+\.)(e[+-]?\d+)? *\z/i === s
  ```
- 21 调整数组顺序使奇数位于偶数前面
  ```
  nums.sort_by { ~_1 & 1 }
  ```
- 35 复杂链表的复制
  ```python
  return deepcopy(head)
  ```
- 40 最小的k个数
  ```python
  return sorted(arr)[:k]
  ```
- 43 1~n整数中1出现的次数
  ```python
  return sum((n+1) // 10**(i+1) * 10**i + min(max((n+1) % 10**(i+1) - 10**i, 0), 10**i) for i in range(10))
  ```
- 53-II 0~n-1中缺失的数字
  ```ruby
  ((0..nums.length).to_a - nums)[0]
  ```
- 55-I 二叉树的深度
  ```ruby
  root ? [max_depth(root.left), max_depth(root.right)].max + 1 : 0
  ```
- 64 求1+2+…+n
  ```ruby
  (1..n).sum
  ```
- 65 不用加减乘除做加法
  ```python
  return add(a, b)
  ```
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
- 48 旋转图像
  ```ruby
  matrix.transpose.map(&:reverse).each_with_index { |l, i| l.each_with_index { |x, j| matrix[i][j] = x } }
  ```
- 65 有效数字
  ```ruby
  /\A[+-]?(\d+|\d+\.\d*|\.\d+)([eE][+-]?\d+)?\z/ === s
  ```
- 66 加一
  ```ruby
  (digits.to_s.gsub(/\D/, "").to_i + 1).to_s.each_char.map(&:to_i)
  ```
- 67 二进制求和
  ```java
  return new BigInteger(a, 2).add(new BigInteger(b, 2)).toString(2);
  ```
- 69 x的平方根
  ```python
  return int(sqrt(x))
  ```
- 70 爬楼梯
  ```ruby
  (1..n).inject([0, 1]) { |(a, b)| [b, a + b] }[1]
  ```
- 74 搜索二维矩阵
  ```ruby
  matrix.bsearch { |x| x[-1] >= target }&.bsearch { |x| x >= target } == target
  ```
- 169 多数元素
  ```ruby
  nums.tally.max_by(&:last).first
  ```
- 283 移动零
  ```python
  nums.sort(key=lambda x: x == 0)
  ```
- 342 4的幂
  ```c
  return n > 0 && !(n & (n - 1)) && !(__builtin_popcount(n - 1) & 1);
  ```
- 344 反转字符串
  ```ruby
  s.reverse!
  ```
- 349 两个数组的交集
  ```python
  return list(set(nums1) & set(nums2))
  ```
- 350 两个数组的交集II
  ```python
  return [a for a, n in (Counter(nums1) & Counter(nums2)).items() for _ in range(n)]
  ```
- 389 找不同
  ```python
  return next(iter(Counter(t) - Counter(s)))
  ```
- 394 字符串解码
  ```ruby
  nil while s.gsub!(/(\d+)\[([^\d\[\]]*)\]/) { $2 * $1.to_i }; s
  ```
- 535 TinyURL的加密与解密
  ```javascript
  encode = decode = x => x
  ```
- 537 复数乘法
  ```ruby
  (num1.sub("+-", "-").to_c * num2.sub("+-", "-").to_c).to_s.sub(/(?<=.)-/, "+-")
  ```
- 819 最常见的单词
  ```ruby
  paragraph.downcase.split(/\W+/).difference(banned).tally.max_by(&:last).first
  ```
- 824 山羊拉丁文
  ```ruby
  sentence.split(" ").each_with_index.map { |x, i| ("aeiouAEIOU".include?(x[0]) ? x : x[1..-1] + x[0]) + "m" + "a" * (i + 2) }.join(" ")
  ```
- 831 隐藏个人信息
  ```ruby
  s.include?("@") ? s.sub(/(\w)\w*(\w)/, "\\1*****\\2").downcase : s.tr("-+() ", "").sub(/\A(\d*)\d{6}(\d{4})\z/) { ($1.empty? ? "" : "+#{"*" * $1.length}-") + "***-***-#{$2}" }
  ```
- 832 翻转图像
  ```python
  return [[1 - x for x in reversed(a)] for a in image]
  ```
- 1742 盒子中小球的最大数量
  ```ruby
  (low_limit..high_limit).each_with_object([0] * 46) { |x, m| m[x.to_s.each_byte.inject(0) { |s, c| s + c - 48 }] += 1 }.max
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

## 极致的丑陋
- 32-II 从上到下打印二叉树II
  ```python
  def levelOrder(self, root: TreeNode) -> List[List[int]]:
      if not root: return []
      ans = [[root]]
      while ans[-1]: ans.append([node for row in [[node.left, node.right] for node in ans[-1]] for node in row if node])
      return [[node.val for node in row] for row in ans][:-1]
  ```

## 很难绷得住
C和Go在做算法题时是难兄难弟，标准库里啥都没有，最长题解必有它俩……

- 047 二叉树剪枝

  递归，优雅！  
  你是一个一个一个带有GC的语言啊啊啊啊啊  
  让我们看看C语言题解：  
  C语言题解直接内存泄漏！
