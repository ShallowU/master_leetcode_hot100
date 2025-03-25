<h1 align="center" style="color:red">leetcode hot 100 trap From 0 base</h1>

## p1 [两数之和](https://leetcode.cn/problems/two-sum/description/?envType=study-plan-v2&envId=top-100-liked)

- 暴力

> vector <int> res(2);// 初始化是（）vector <int> res(2,-1);//两个元素都是赋值-1
>
> 还有多个数返回可以return {i，j}   return{}

- 哈希表

```c++
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        map<int,int> test;
        for(int i=0;i<nums.size();i++)
            test.insert({nums[i],i});// 这里是插入键值对pair组合即{}
            // 也可以用[] test[nums[i]]=i;
        for(int i=0;i<nums.size();i++)
            if(test.find(target-nums[i])!=test.end()&&test[target-nums[i]]!=i)
            {
                map<int,int>::iterator iter=test.find(target-nums[i]);
                // 类型要么map<int,int>::iterator 要么auto
                int j=iter->second;
                return{i,j};
            }
        return{};
    }
};

// 注意原数组中重复的元素放到map中由于键是唯一的，所以其值会被后一个相同值的下标替代
// 所以遍历时候要用原数组，并且只要找到元素不能是自己

class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int,int> mp;
        for(int i=0;i<nums.size();i++)
        {
            mp[nums[i]]=i;
        }
        for(int i=0;i<nums.size();i++)	// 重要
        {
            auto aa=mp.find(target-nums[i]);
            if(aa!=mp.end()&&aa->second!=i)	//因为不会是自己了
                return{i,aa->second};
        }
        return {};
        
    }
};
```

>map<two type> name;初始化
>
>添加元素要么用insert（记得插入{}要么用[ ]
>
>.find函数是用来查找键的，.begin() and .end()
>
>类型要么map<int,int>::iterator 要么auto
>
>it->first and it->second 或者test[target-nums[i]]!=i这样引用值
>
>`.count()` 是 C++ 中 `map` 或 `unordered_map` 等容器的成员函数，用来查找某个键（key）在容器中出现的次数。
>
>对于 `map` 或 `unordered_map`，每个键最多只出现一次，因此 `.count()` 的返回值要么是 `0`（表示没有找到该键），要么是 `1`（表示找到了该键）。但对于 `multimap` 和 `unordered_multimap` 等容器，`count()` 会返回该键出现的次数。
>
> multimap<int, int> myMultimap;    
>
>myMultimap.insert({1, 10});
>
>//迭代器刪除
>iter = mapStudent.find("123");
>mapStudent.erase(iter);
>
>//用关键字刪除
>int n = mapStudent.erase("123"); //如果刪除了會返回1，否則返回0
>
>//用迭代器范围刪除 : 把整个map清空
>mapStudent.erase(mapStudent.begin(), mapStudent.end());
>//等同于mapStudent.clear()
>
>// 整体思路
>
>键值对交换，将需要查找的放到键上，即得到对应的值
>
>当对当前得nums[i]匹配是否存在target-nums[i]这个，如果找到就可以结束得到对应得值了

## p2 [字母异位词分组](https://leetcode.cn/problems/group-anagrams/?envType=study-plan-v2&envId=top-100-liked)

- 哈希表

```
// 思路
// 通过将每个string词进行sort排序，这样就可以将同类型的词语有一个共同的标识，再进行unordered_map<string,vector<string>> mp,构建哈希表，每个标识对应一个string列表类即可
// mp[key].push_back(strs[i]) 注意插入用push_back

```

```c++
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        unordered_map<string,vector<string>> mp;
        for(int i=0;i<strs.size();i++) //for(string &str:strs)
        {
            string key=strs[i];
            sort(key.begin(),key.end());
            mp[key].push_back(strs[i]);
        }
        vector<vector<string>> ans;
        for(auto it=mp.begin();it!=mp.end();it++)
        {
            ans.push_back(it->second);
        }
        return ans;
        
    }
};
```

>// unordered_map is faster than map
>
>// unordered_map is implemented by hash table
>
>// map is implemented by red-black tree
>
>map是按键进行了排序，所以更消耗空间，而unordered_map是哈希值随机的
>
>emplace_back(构造参数列表)只调用一次构造函数，而push_back(构造参数列表)会调用一次构造函数+一次移动构造函数。其余情况下，二者相同。`emplace_back` 来提高效率，特别是当你要构造复杂对象时。

## p3[最长连续序列](https://leetcode.cn/problems/longest-consecutive-sequence/description/?envType=study-plan-v2&envId=top-100-liked)



```c++
// 第一种方法：使用并查集
class Solution {
public:

    unordered_map<int, int> parent; //使用哈希表进行存储父节点表示连续子序列
    unordered_map<int, int> size; //size存储每个序列的长度
    //find查找父节点
    int find(int x) {
        if (x != parent[x]) {
            parent[x] = find(parent[x]);// 压缩路径
        }
        return parent[x];
    }
    //合并节点为同一个父节点
    void Union(int x, int y) {
        int root1 = find(x);
        int root2 = find(y);
        if (root1 == root2) return; //相同父节点则可以直接跳过
        //使用秩更小的进行节点union 
        if (size[root1] < size[root2]) swap(root1, root2);
        parent[root2] = root1;
        size[root1] += size[root2]; //相应序列长度进行相加
    }
    int longestConsecutive(vector<int>& nums) {
    // 初始化parent和size
    for (int num : nums) {
        parent[num]=num; //初始化其父节点为其本身
        size[num]=1;
    }
    for (int num : nums) {
        // 双向寻找连续子序列
        if (parent.count(num - 1)) {
            Union(num, num - 1);
        }
        if (parent.count(num + 1)) {
            Union(num, num + 1);
        }
    }
    int maxSize = 0;
    for (int num : nums) {
        maxSize = max(maxSize, size[find(num)]); //最找最大连续列的长度
    }
    return maxSize;
    }
};

// 用set查找
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        unordered_set<int> nset(nums.begin(),nums.end());
        int ans=0;
        int foot=0;
        for(const int& num:nset)
        {for(const int& num:nset)
            int currentnum=num;
            if(!nset.count(currentnum-1))
            {
                foot++;
                while(nset.count(currentnum+1))
                {
                    foot++;
                    currentnum=currentnum+1;
                }
                ans=max(ans,foot);
                foot=0;
            }
        }
        return ans;
    }
};

// 自己用的map
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        if(nums.size()==0)
            return 0;
        map<int,int> arr;
        for(int i=0;i<nums.size();i++)
        {
            arr[nums[i]]=1;
        }
        int maxlen=1;
        int tmp=1;
        for(auto it=arr.begin();it!=arr.end();it++)
        {
            if(arr.find(it->first-1)!=arr.end())
                continue;
            else
            {
                while(arr.find(it->first+1)!=arr.end())
                {
                    tmp++;
                    it=arr.find(it->first+1);
                }
                if(tmp>maxlen)
                    maxlen=tmp;
            }
            tmp=1;

        }
        return maxlen;
    }
};
```

> 核心需要掌握先去除重复元素，再找当前元素-1的值是否在数组中存在，如果有的话，那就当前这个不用算步长了（直接continue，没有的话就当第一个算
>
> unordered_set<int> nset(nums.begin(),nums.end());//学习初始化方式
>
> for(const int& num:nset) //遍历方式
>
> 或者for(int num:nset)

## P4[移动0到数组末尾](https://leetcode.cn/problems/move-zeroes/description/?envType=study-plan-v2&envId=top-100-liked)

```c++
// 双指针两次遍历
// 思想：就是将非零的元素挨个往数组首放，一个个的放，最后末尾剩出来的位置就全放0
class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        int nonindex=0;
        for(int i=0;i<nums.size();i++)
        {
            if(nums[i])
            {
                nums[nonindex]=nums[i];
                nonindex++;
            }
        }
        for(int i=nonindex;i<nums.size();i++)
        {
            nums[i]=0;
        }
    }
};

// 单次遍历
// 思想：从一个位置开始，如果元素不为0就与前面一个j下标标记元素交换 ，这样逐渐0会换到最后
class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        int j=0;
        for(int i=0;i<nums.size();i++){
            if(nums[i])
            {
                swap(nums[i],nums[j]);
                j++;
            }
        }
    }
};
```

## p5[矩阵接雨水最大](https://leetcode.cn/problems/container-with-most-water/description/?envType=study-plan-v2&envId=top-100-liked)

```c++
// 暴力
// 过不了所有测试用例
class Solution {
public:
    int maxArea(vector<int>& height) {
        int maxans=0;
        for(int i=0;i<height.size();i++)
        {
            for(int j=i+1;j<height.size();j++)
            {
                int high=min(height[i],height[j]);
                int ans=(j-i)*high;
                if(ans>maxans) maxans=ans;
            }
        }
        return maxans;
    }
};
// 利用双指针，left和right为左右边界
// 我们left++和right--都是为了尝试取到更多的水，如果短的板不动的话，取到的水永远不会比上次多。
class Solution {
public:
    int maxArea(vector<int>& height) {
        int maxans=0;
        int left=0,right=height.size()-1;
        while(left<right)
        {
            int ans=(right-left)*min(height[left],height[right]);
            maxans=max(ans,maxans);
            if(height[left]<=height[right])
            {
                left++;
            }
            else    right--;
        }
        return maxans;
    }
};
// 能盛多少水由短板决定，抛弃最黑暗的自己，才能有未来
```

## p6[三数之和为0](https://leetcode.cn/problems/3sum/?envType=study-plan-v2&envId=top-100-liked)

```c++
// 先使用排序再固定第一个数，剩下使用双指针遍历
// 最重要的是去重
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        int n=nums.size();
        vector<vector<int>> ans;
        sort(nums.begin(),nums.end()); //先排序从小到大
        if(n<3||nums[0]>0) return{};  // 如果数目都没有3个或者排序后第一个数都大于0 直接返回
        for(int i=0;i<n;i++)
        {
            if(i>0&& nums[i]==nums[i-1])	// 去重第一个数重复
                continue;
            int left=i+1,right=n-1;         // 固定第一个数后两边的双指针区间
            while(left<right)
            {
                if(nums[i]+nums[left]+nums[right]==0)
                {
                    ans.push_back({nums[i],nums[left],nums[right]});
                    while(left<right&&nums[left]==nums[left+1])		// 确保答案第二个数去重
                        left++;
                    while(left<right&&nums[right]==nums[right-1])	// 确保答案第三个数去重
                        right--;
                    left++;		// 到达新的数，由于left+right已经获得答案了，则left和right都要更新
                    right--;	// left+【right-1】明显不能满足答案了
                }
                else if(nums[i]+nums[left]+nums[right]<0)	
                    left++;
                else
                    right--;
            }
        }
        return ans;
        
    }
};
```



## p7[接雨水](https://leetcode.cn/problems/trapping-rain-water/?envType=study-plan-v2&envId=top-100-liked)

```c++
// 动态规划
// 思路需要想到如何算每个柱子的雨水
//对于下标 i，下雨后水能到达的最大高度等于下标 i 两边的最大高度的最小值，下标 i 处能接的雨水量等于下标 i 处的水能到达的最大高度减去 height[i]。
// 创建两个长度为 n 的数组 leftMax 和 rightMax。对于 0≤i<n，leftMax[i] 表示下标 i 及其左边的位置中，height 的最大高度，rightMax[i] 表示下标 i 及其右边的位置中，height 的最大高度。

class Solution {
public:
    // 思路：当前柱子的水量为此柱子左边最高柱子和右边最高柱子的小值减去height【i】
    // 即water[i]= min(leftmax,rightmax)-height[i]
    // 运用动态规划逐个求每个i的leftmax(<=i),rightmax(>i)
    int trap(vector<int>& height) {
        if(height.size()==0)    return 0;
        int left=0,right=height.size()-1;
        vector<int> leftmax(right+1);
        vector<int> rightmax(right+1);
        leftmax[0]=height[0];
        rightmax[right]=height[right];
        int ans=0;
        for(int i=1;i<height.size();i++)
        {
            leftmax[i]=max(leftmax[i-1],height[i]);
        }
        for(int j=right-1;j>=0;j--)
        {
            rightmax[j]=max(rightmax[j+1],height[j]);
        }
        for(int i=0;i<height.size();i++)
        {
            ans+=min(leftmax[i],rightmax[i])-height[i];
        }
        return ans;
    }
};

// 双指针优化上述leftmax和rightmax的空间数组值
可算看懂了，原来双指针同时开两个柱子接水。大家题解没说清楚，害得我也没看懂。 对于每一个柱子接的水，那么它能接的水=min(左右两边最高柱子）-当前柱子高度，这个公式没有问题。同样的，两根柱子要一起求接水，同样要知道它们左右两边最大值的较小值。

问题就在这，假设两柱子分别为 i，j。那么就有 iLeftMax,iRightMax,jLeftMx,jRightMax 这个变量。由于 j>i ，故 jLeftMax>=iLeftMax，iRigthMax>=jRightMax.

那么，如果 iLeftMax>jRightMax，则必有 jLeftMax >= jRightMax，所有我们能接 j 点的水。

如果 jRightMax>iLeftMax，则必有 iRightMax >= iLeftMax，所以我们能接 i 点的水。

而上面我们实际上只用到了 iLeftMax，jRightMax 两个变量，故我们维护这两个即可。                                                             
class Solution {
public:
    int trap(vector<int>& height) {
        if(height.size()==0)    return 0;
        int left=0,right=height.size()-1;
        int leftmax=0,rightmax=0;
        int ans=0;
        while(left<right)
        {
            leftmax=max(leftmax,height[left]);
            rightmax=max(rightmax,height[right]);
            ans+=leftmax<rightmax? leftmax-height[left++]:rightmax-height[right--] ;
        }
        return ans;
    }
};
// 注意 while 循环可以不加等号，因为在「谁小移动谁」的规则下，相遇的位置一定是最高的柱子，这个柱子是无法接水的。

```

## p8[无重复字符的最长字串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/description/?envType=study-plan-v2&envId=top-100-liked)

```c++
// 首先是我的错误思路，只想到一半
// "dvdf"这种就是错的，我的代码答案为2不是正确的3、
// 错误代码：
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        if(s.size()==0) return 0;
        if(s.size()==1) return 1;
        int maxs=0;
        int curs=0;
        set<char> subs;
        subs.insert(s[0]);
        curs++;
        for(int i=1;i<s.size();i++)
        {
            if(subs.find(s[i])==subs.end())
            {
                subs.insert(s[i]);
                curs++;
                maxs=max(curs,maxs);
            }
            else
            {
                curs=1;
                maxs=max(curs,maxs);
                subs.clear();
                subs.insert(s[i]);
            }
        }
        return maxs;
        
    }
};
// 错误原因：思路是从头扫描第一个字符，dv遇到d时候就舍弃了左边界全部从头开始了，所以最终是2，而正确思路是不断丢掉左边界的元素
// 直到区间没有重复元素，比如dv遇到d时候就丢掉左边d，剩下vd，再遇到f就是vdf。
// 我的错误思路是vd遇到d就直接从d从头开始

模板：
//外层循环扩展右边界，内层循环扩展左边界
for (int l = 0, r = 0 ; r < n ; r++) {
	//当前考虑的元素
	while (l <= r && check()) {//区间[left,right]不符合题意
        //扩展左边界
    }
    //区间[left,right]符合题意，统计相关信息
}

class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        if(s.size()==0) return 0;
        unordered_set<char> subs;	// unordered_set效率比set更高效率不排序
        int n=s.size();	//可以后续减少size的打字*_*
        int ans=0;
        for(int left=0,right=0;right<n;right++)	//外层循环处理右边界
        {
            char ch=s[right];
            while(subs.count(ch))	// 处理不符合题意的左边界，不断收缩左边界使不含重复元素
            {
                subs.erase(s[left++]);
            }
            subs.insert(ch);	// 符合题意不断insert
            ans=max(ans,right-left+1);
        }
        return ans;
    }
};
```



## p9[找到字符串中所有的字母异位词](https://leetcode.cn/problems/find-all-anagrams-in-a-string/description/?envType=study-plan-v2&envId=top-100-liked)

```c++
// 错误解法，思路是一部分可以通过 50/65过，这里是用的set而不是multiset，会使
s="ababababab"
p="aab"
有重复元素的检测错误，但即使继续用multiset也会很麻烦处理不了，只能放弃了呜呜呜呜
看答案解题思路

class Solution {
public:
    vector<int> findAnagrams(string s, string p) {
        int ns=s.size(),np=p.size();
        if(np>ns)   return {};
        set<char> str;
        for(int i=0;i<np;i++)
        {
            str.insert(p[i]);
        }
        set<char> str1(str);
        int left=0,right=0;
        vector<int> ans;
        for(;left<=ns-np;left++)
        {
            int tmp=left;
            char ch=s[left];
            right=left+np-1;
            while(left<=right&&str.count(s[left]))
            {
                auto it=str.find(s[left]);
                str.erase(it);
                // str.erase(s[left]);
                left++;
                if(str.empty()) ans.push_back(tmp);
            }
            left=tmp;
            str.clear();
            for(auto it:str1)
                str.insert(it);
        }
        return ans;
    
    }
};

// 正确思路
// 归根揭底就是看这个np滑动区间中各个字母出现的次数是否相等，判断两个统计字母次数的vector是否相等
// 用字符的ascii-a，范围在0-25(常见的字符统计技巧)
// 每到一个np长度区间就进行比较，结束后都进行right++ left++到一个新区间
class Solution {
public:
    vector<int> findAnagrams(string s, string p) {
        int ns=s.size(),np=p.size();
        if(np>ns)   return {};
        vector<int> ans;
        vector<int> scount(26,0),pcount(26,0);
        for(auto it:p)  pcount[it-'a']++;
        int left=0,right=0;
        for(;right<ns;right++)
        {
            scount[s[right]-'a']++;
            if(right-left+1==np)
            {
                if(scount==pcount)  ans.push_back(left);
                scount[s[left]-'a']--; // 记得减掉移走的left对应字母次数一次
                left++;
            }
        }
        return ans;
    }
};
```

## p10[和为k的子数组](https://leetcode.cn/problems/subarray-sum-equals-k/description/?envType=study-plan-v2&envId=top-100-liked)

```c++
1 <= nums.length <= 2 * 10^4
-1000 <= nums[i] <= 1000
-107 <= k <= 107    
// 例子中可能有负数
// 这种下面暴力做法差最后一个示例，偶尔会超时 
// 思路：就是枚举每一个值作为right，然后再遍历left中，试试加起来有值等于是否
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        if(nums.size()==0) return 0;
        int n=nums.size();
        int ans=0;
        for(int i=0;i<n;i++)
        {
            int sum=0;
            for(int j=i;j>=0;j--)  //for (int end = start; end < nums.size(); end++) 从左右遍历也行，以i为起始
            {
                sum+=nums[j];
                if(sum==k)
                    ans++;
            }
        }
        return ans;
    }
};

//
// 定义 pre[i] 为 [0..i] 里所有数的和，则 pre[i] 可以由 pre[i−1] 递推而来，即：
pre[i]=pre[i−1]+nums[i]
那么「[j..i] 这个子数组和为 k 」这个条件我们可以转化为
pre[i]−pre[j−1]==k
简单移项可得符合条件的下标 j 需要满足
pre[j−1]==pre[i]−k
所以我们考虑以 i 结尾的和为 k 的连续子数组个数时只要统计有多少个前缀和为 pre[i]−k 的 pre[j] 即可。
//
前缀和的概念
首先，我们使用一个叫做“前缀和”的概念。对于数组中的任何位置 j，前缀和 pre[j] 是数组中从第一个元素到第 j 个元素的总和。这意味着如果你想知道从元素 i+1 到 j 的子数组的和，你可以用 pre[j] - pre[i] 来计算。
使用 Map 来存储前缀和
在代码中，我们用一个 Map（哈希表）来存储每个前缀和出现的次数。这是为了快速检查某个特定的前缀和是否已经存在，以及它出现了多少次。
核心逻辑解析
当我们在数组中向前移动时，我们逐步增加 pre（当前的累积和）。对于每个新的 pre 值，我们检查 pre - k 是否在 Map 中：
pre - k 的意义：这个检查的意义在于，如果 pre - k 存在于 Map 中，说明之前在某个点的累积和是 pre - k。由于当前的累积和是 pre，这意味着从那个点到当前点的子数组之和恰好是 k（因为 pre - (pre - k) = k）。
如何使用这个信息：如果 pre - k 在 Map 中，那么 pre - k 出现的次数表示从不同的起始点到当前点的子数组和为 k 的不同情况。这是因为每一个 pre - k 都对应一个起点，使得从那个起点到当前点的子数组和为 k。
因此，每当我们找到一个 pre - k 存在于 Map 中时，我们就把它的计数（即之前这种情况发生的次数）加到 count 上，因为这表示我们又找到了相应数量的以当前元素结束的子数组，其和为 k。
 class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        if(nums.size()==0) return 0;
        int ans=0;
        int pre=0;
        unordered_map<int,int> mp;
        mp[0]=1; //为什么这样初始化？起始使防止pre[i]==k，区间全部就满足条件，这样就少加了一次
        for(auto num:nums)
        {
            pre+=num;
            if(mp.find(pre-k)!=mp.end())
            {
                ans+=mp[pre-k];
            }
            mp[pre]++;	// mp[pre]的值可能会有相同的，但是次数会增加
        }
        return ans;
    }
};


    
```

## p11[滑动窗口最大值](https://leetcode.cn/problems/sliding-window-maximum/description/?envType=study-plan-v2&envId=top-100-liked)

```c++
// 暴力做法，复杂度O(nk)，不能过
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        vector<int> ans;
        for(int i=0;i<=nums.size()-k;i++)
        {
            int j=i;
            int maxnum=nums[i];
            for(;j<=i+k-1;j++)
            {
                maxnum=max(maxnum,nums[j]);
            }
            ans.push_back(maxnum);
        }
        return ans;
    }
};

// 使用multiset迭代器运用类似暴力的做法
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        if(nums.size()==0) return{};
        vector<int> ans;
        multiset<int> b(nums.begin(),nums.begin()+k);	//multiset有自动排序，且允许重复元素
        // 该区间左闭由开，共k个元素，但只到nums[k-1]
        // 初始化的迭代器指针
        c.begin() 返回一个迭代器，它指向容器c的第一个元素

        c.end() 返回一个迭代器，它指向容器c的最后一个元素的下一个位置

        c.rbegin() 返回一个逆序迭代器，它指向容器c的最后一个元素

        c.rend() 返回一个逆序迭代器，它指向容器c的第一个元素前面的位置            
        //
        ans.push_back(*b.rbegin()); //最后一个元素就是最大的
        for(int i=1;i<nums.size()-k+1;i++)
        {
            b.erase(b.find(nums[i-1])); // 删掉上一个窗口的首值
            b.insert(nums[i+k-1]);		// 插入新值
            ans.push_back(*b.rbegin()); // 最大值
        }
        return ans;
    }
};

// 使用优先队列
https://blog.csdn.net/albertsh/article/details/108552268
优先队列：本质是队列，但加上了排序，所以同时又是一个堆，默认构造是大顶堆，先出最大的元素，
 队列有：push（） pop() top() emplace() empity() size()
  priority_queue<int>==priority_queue<int,vector<int>,less<int>>; //大根堆
priority_queue<int,vector<int>,greater<int>>;// 小根堆
  经典优先队列问题就是数组前k大的数
思想：类似裁员广进，每次出最大的数，但要判断该值是否在区间内。第一时间进入的值，不会删除旧的不在区间的值，会随着每次
    pop判断是否是以前的区间值。所以需要下标就pair<int ,int>
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        if(nums.size()==0) return{};
        vector<int> ans;
        priority_queue<pair<int,int>> q;
        for(int i=0;i<k;i++)
        {
            q.push(make_pair(nums[i],i));	//q.emplace(nums[i], i);直接构造
        }
        ans.push_back(q.top().first); // 第一个窗口最大值
        for(int i=k;i<nums.size();i++)
        {
            q.push(make_pair(nums[i],i)); //先加入
            while(q.top().second<=i-k)	// 判断当前最大值是否是以前区间的值
            {
                q.pop();
            }
            ans.push_back(q.top().first);
        }
        return ans;

    }
};
// 单调队列（优先队列）
我悟了，队尾只要有更年轻（i越靠后）同时还能力更强（数值越大）的，留着其他比它更早入职同时能力却更差的没有什么意义，统统开了；队首的虽然能力最强，但是年龄最大，一旦发现它超过年龄范围（不在滑动窗口的范围内），不用看能力就可以直接开了。
 //
```

## p12[数组中第k个大元素](https://leetcode.cn/problems/kth-largest-element-in-an-array/description/?envType=study-plan-v2&envId=top-100-liked)

```c++
class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        // 前k大就用小根堆装k个，最后剩的就是前面k大的元素，第一个top就是第k大
        // 反之前k小就用大根堆
        priority_queue<int,vector<int>,greater<int>> q;
        for(auto n:nums)
        {
            if(q.size()==k)
            {
                if(n>q.top())
                {
                    q.pop();
                    q.push(n);
                }
            }
            else
                q.push(n);
        }
        int ans=0;
        ans=q.top();
        return ans;
    }
};
```

## p13[前k个高频元素](https://leetcode.cn/problems/top-k-frequent-elements/description/?envType=study-plan-v2&envId=top-100-liked)

```c++
// 反正看到前k大就用优先队列，这里统计次数用map，放入优先队列时候放次数在前面，作为排序用次数排序，
// 用默认的大根堆，先输出频率大的对应元素
// 首先，这段代码是在遍历一个unordered_map<int, int>的mp。循环里用的是for(auto it:mp)，这样it应该是一个pair的副本，也就是每个元素是一个键值对的拷贝。这时候访问成员的写法应该是用点运算符，比如it.first和it.second，对吧？因为在这种情况下，it是一个对象，而不是指针或者迭代器。所以这里用it.second和it.first是正确的。
// 那如果换成it->second的话，那应该是当it是一个指针或者迭代器的时候才能用。比如，如果循环写成for(auto it = mp.begin(); it != mp.end(); it++)，这时候it是一个迭代器，指向键值对，所以需要用->来访问成员，比如it->first和it->second。但是原来的代码中用的是范围for循环，所以每个it是一个pair类型的对象，而不是指针或者迭代器，所以应该用点操作符。
// 7ms
class Solution {
public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        int n=nums.size();
        if(n==0) return {};
        vector<int> ans;
        unordered_map<int,int> mp;
        for(auto n:nums)
        {
            if(mp.find(n)!=mp.end())
            {
                mp[n]++;
            }
            else
            {
                mp[n]=1;
            }
        }
        priority_queue<pair<int,int>> q;
        for(auto it=mp.begin();it!=mp.end();it++)
        {
            q.emplace(it->second,it->first); // 
        }
        int count=1;
        while(!q.empty())
        {
            ans.push_back(q.top().second);
            q.pop();
            count++;
            if(count>k)
                break;
        }
        return ans;
    }
};
// 用前k大元素的方式进行，这样这里用的小根堆
// priority_queue<pair<int,int>,vector<pair<int,int>>,greater<pair<int,int>>> q;
// pair 格式书写！！！！！
// 这样复杂度只有一半了 3 ms
class Solution {
public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        int n=nums.size();
        if(n==0) return {};
        vector<int> ans;
        unordered_map<int,int> mp;
        for(auto n:nums)
        {
            if(mp.find(n)!=mp.end())
            {
                mp[n]++;
            }
            else
            {
                mp[n]=1;
            }
        }
        priority_queue<pair<int,int>,vector<pair<int,int>>,greater<pair<int,int>>> q;
        for(auto it=mp.begin();it!=mp.end();it++)
        {
            if(q.size()==k)
            {
                if(it->second>q.top().first)
                {
                    q.pop();
                    q.emplace(it->second,it->first);
                }
            }
            else
                q.emplace(it->second,it->first);
        }
        while(!q.empty())
        {
            ans.push_back(q.top().second);
            q.pop();
        }
        return ans;
    }
};
```

## p14[最大子数组和](https://leetcode.cn/problems/maximum-subarray/description/?envType=study-plan-v2&envId=top-100-liked)

```c++
// 使用动态规划
// 我们用 f(i) 代表以第 i 个数结尾的「连续子数组的最大和」，那么很显然我们要求的答案就是：
因此我们只需要求出每个位置的 f(i)，然后返回 f 数组中的最大值即可。那么我们如何求 f(i) 呢？我们可以考虑 nums[i] 单独成为一段还是加入 f(i−1) 对应的那一段，这取决于 nums[i] 和 f(i−1)+nums[i] 的大小，我们希望获得一个比较大的，于是可以写出这样的动态规划转移方程：
f(i)=max{f(i−1)+nums[i],nums[i]}

class Solution {
public:
    int maxSubArray(vector<int>& nums) 
    {
        vector<int> maxi(nums);//以i结尾索引数组的连续子数组最大和
        for(int i=1;i<nums.size();i++)
        {
            maxi[i]=max(nums[i],maxi[i-1]+nums[i]);
        }
        return *max_element(maxi.begin(),maxi.end());//快速找迭代器最大值最小值
    }
};

// 使用前缀和
class Solution {
public:
    int maxSubArray(vector<int>& nums) 
    {
        // 使用前缀和，区间的和等于两个前缀和相减
        // 所以区间和最大值等于当前前缀和-前面最小前缀和
        // 初始最小前缀和为0
        int ans=INT_MIN;
        int pre_num=0;
        int min_pre_num=0;
        for(auto x:nums)
        {
            pre_num+=x; // 当前前缀和
            ans=max(pre_num-min_pre_num,ans); // 当前前缀和减去前缀和的最小值
            min_pre_num=min(pre_num,min_pre_num); // 更新最小前缀和
        }
        return ans;
    }
};
```

## p15[合并区间](https://leetcode.cn/problems/merge-intervals/description/?envType=study-plan-v2&envId=top-100-liked)

```c++
题目：
以数组 intervals 表示若干个区间的集合，其中单个区间为 intervals[i] = [starti, endi] 。请你合并所有重叠的区间，并返回 一个不重叠的区间数组，该数组需恰好覆盖输入中的所有区间 。
示例 1：
输入：intervals = [[1,3],[2,6],[8,10],[15,18]]
输出：[[1,6],[8,10],[15,18]]
解释：区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
// 先进行排序，让一维vector的按左端点值从小到大排序，便于后续比较和跳过
// 如果后一个一维vector的左端点值在当前寻找的一维vector区间中，并且一维vector的右端点值大于当前寻找的一维vector右端点值
// 则可以选择一个maxrightnum=max(maxrightnum,intervals[j][jright]);
 //   最后push{intervals[i][0],maxrightnum}，如果没有满足上述条件就push本身

class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        sort(intervals.begin(),intervals.end());
        vector<vector<int>> ans;
        for(int i=0;i<intervals.size();i++)
        {
            vector<int>tmp(intervals[i]);
            int skipcount=1; // 中间一些小区间包含于当前寻找的区间，后面可以直接跳过
            int iright=intervals[i].size()-1;
            int maxrightnum=intervals[i][iright];
            vector<int> res;
            for(int j=i+1;j<intervals.size();j++)
            {
                int jright=intervals[j].size()-1;
                if(intervals[j][0]<=maxrightnum) // 一直遍历左端点是否小于当前融合区间的右端点（即最大值
                {
                    maxrightnum=max(maxrightnum,intervals[j][jright]);
                    skipcount++;
                }
                else
                    break;   
            }
            if(skipcount==1)
                ans.push_back(tmp); // 没有合并，push本身
            else
                ans.push_back({intervals[i][0],maxrightnum});
            i+=skipcount-1;
        }
        return ans;
    }
};
```

## p16[轮转数组](https://leetcode.cn/problems/rotate-array/description/?envType=study-plan-v2&envId=top-100-liked)

```c++
//  第一想到做法就是找规律，但要多开销一个数组
class Solution {
public:
    void rotate(vector<int>& nums, int k) {
        k=k%nums.size();
        int differ=nums.size() -k;
        vector<int> tmp;
        for(int i=differ;i<nums.size();i++)
        {
            tmp.push_back(nums[i]);
        }
        for(int i=0;i<differ;i++)
        {
            tmp.push_back(nums[i]);
        }
        for(int i=0;i<nums.size();i++)
        {
            nums[i]=tmp[i];
        }
    }
};

// o(1)的空间复杂度，但是超时了，差最后一个测试样例
class Solution {
public:
    void rotate(vector<int>& nums, int k) {
        k=k%nums.size();
        while(k>0)
        {
            int x=nums[nums.size()-1];
            for(int i=nums.size()-1;i>=1;i--)
            {
                nums[i]=nums[i-1];
            }
            nums[0]=x;
            k--;
        }
    }
};
```

## p17[除自身以外数组的乘积](https://leetcode.cn/problems/product-of-array-except-self/?envType=study-plan-v2&envId=top-100-liked)

```c++
class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) {
        // 类似分治的思想
        // 乘积等于左边所有数的乘积和右边所有数的乘积再乘积
        // 先求出左右数的乘积保存起来，以空间换时间
        int n=nums.size();
        vector<int> l(n),r(n);
        vector<int> ans(n);
        l[0]=1;
        for(int i=1;i<n;i++)
        {
            l[i]=l[i-1]*nums[i-1];
        }
        r[n-1]=1;
        for(int i=n-2;i>=0;i--)
        {
            r[i]=r[i+1]*nums[i+1];
        }
        for(int i=0;i<n;i++)
        {
            ans[i]=l[i]*r[i];
        }
        return ans;

    }
};
```

## p18[矩阵置0](https://leetcode.cn/problems/set-matrix-zeroes/description/?envType=study-plan-v2&envId=top-100-liked)

```c++
class Solution {
public:
    void setZeroes(vector<vector<int>>& matrix) {
        // 本质上进行标记，标记有0出现的行和0出现的列
        // 然后遍历原数组，如果该元素位于该行或者该列那就置0
        int m=matrix.size();
        int n=matrix[0].size();
        vector<int> row(m,0),col(n,0);
        for(int i=0;i<m;i++)
        {
            for(int j=0;j<n;j++)
            {
                if(matrix[i][j]==0)
                {
                    row[i]=col[j]=1;
                }
            }
        }
        for(int i=0;i<m;i++)
        {
            for(int j=0;j<n;j++)
            {
                if(row[i]||col[j])
                    matrix[i][j]=0;
            }
        }
    }
};
```

## p18[螺旋矩阵](https://leetcode.cn/problems/spiral-matrix/description/?envType=study-plan-v2&envId=top-100-liked)

```c++
// 首先这道可以学到很多，值得深挖多种解决方案
// 应用方向数组进行解决这非常重要，每遇到边界以外或者以及visit的元素就顺时针调整一下方向
// int direct[4][2]={{0,1},{1,0},{0,-1},{-1,0}};4个标记方向，使用index来转变，第一个元素是行，第二个是列，首先{0,1}是行不变，col+1，同理
// {1,0}是row+1，col不变，
// 判断是否所有结束了，就是共添加了m*n元素就是
// 为了避免回到以前添加过的元素，需要设置visted数组进行标记，下一个元素是visted过的，则要进行顺时针旋转了

class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        int m=matrix.size();
        int n=matrix[0].size();
        vector<int> ans; // vector<int> ans(m*n)
        int direct[4][2]={{0,1},{1,0},{0,-1},{-1,0}};
        vector<vector<int>> visited(m,vector<int>(n,0));
        // or int visited[m][n];
        //  memset(visited,0,sizeof(visited))
         
        int total=m*n;

        int row=0,col=0;
        int idex=0;
        for(int i=0;i<total;i++)
        {
            ans.push_back(matrix[row][col]); // ans[i]=matrix[row][col]
            visited[row][col]=1; // 经过以后设置为1
            int nextrow=row+direct[idex][0];
            int nextcol=col+direct[idex][1];
            // 数组界限都是0-m-1，0-n-1，所以遇到m或者n就是越界了，！！！！
            if(nextrow<0||nextrow>=m||nextcol<0||nextcol>=n||visited[nextrow][nextcol]==1)
                idex=(idex+1)%4;// 要mod4，因为index会循环，大于4就灭有意义了
            row+=direct[idex][0];
            col+=direct[idex][1];

        }
        return ans;
        
    }
};
```

## p19[旋转图像](https://leetcode.cn/problems/rotate-image/description/?envType=study-plan-v2&envId=top-100-liked)

```c++
// 找规律，会发现旋转后的位置与之前的行 列相关
// 具体是matrix[i][j] -> matrix_new[j][n - i - 1],但这里开辟了新空间而不是原地旋转

class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        int n = matrix.size();
        // C++ 这里的 = 拷贝是值拷贝，会得到一个新的数组
        auto matrix_new = matrix;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                matrix_new[j][n - i - 1] = matrix[i][j];
            }
        }
        // 这里也是值拷贝
        matrix = matrix_new;
    }
};

// 原地旋转
// 先进行转置再逆序
class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        int n=matrix.size();
        //  矩阵转置
        for(int i=0;i<n;i++)
        {
            for(int j=0;j<i;j++) // 转置下标是将倒三角的一半进行对换
            {
                swap(matrix[i][j],matrix[j][i]);
            }
        }
        // 每行对称交换或者逆序
        // for(int i=0;i<n;i++)
        // {
        //     reverse(&matrix[i][0],&matrix[i][0]+n);
        // }
        for(auto &it:matrix)
            reverse(it.begin(),it.end());
    }
};

使用 for(auto &it: matrix)：it 是引用，修改 it 会修改 matrix 中的原始数据。

使用 for(auto it: matrix)：it 是副本，修改 it 不会影响 matrix 中的原始数据。
    如果是将it指向的值添加到其他位置或者容器中，两者选择均可，如果原地修改只能是&
```

## p20[岛屿数量](https://leetcode.cn/problems/number-of-islands/description/?envType=study-plan-v2&envId=top-100-liked)

```c++
// 很传统的dfs，但是这里是网状结构而不是二叉树
void dfs(原二维数组，visited[]，r，c)
{
    判断r ，c是否超过边界
        return；
        
     判断是否visited
        return；
        
      visited[r][c]=1;
    
    if(该二维数组元素是我们需要判断的值)
    {
		上下左右
         四个方向进行递归dfs（）
    }
    
}

class Solution {
public:
    void dfs(vector<vector<char>>& grid,vector<vector<int>> &visited,int r,int c)
    {
        int nr=grid.size();
        int nc=grid[0].size();
        if(r<0||r>=nr||c<0||c>=nc)
            return;
        if(visited[r][c]==1)
            return;
        visited[r][c]=1;
        if(grid[r][c]=='0')
            return;
        dfs(grid,visited,r-1,c);
        dfs(grid,visited,r+1,c);
        dfs(grid,visited,r,c+1);
        dfs(grid,visited,r,c-1);
    }
    int numIslands(vector<vector<char>>& grid) {
        int nr=grid.size();
        int nc=grid[0].size();
        vector<vector<int>> visited(nr,vector<int>(nc,0));
        int ans=0;
        for(int i=0;i<nr;i++)
        {
            for(int j=0;j<nc;j++)
            {
                if(grid[i][j]=='1'&&visited[i][j]==0)
                {
                    ans++;
                    dfs(grid,visited,i,j);
                }
            }
        }
        return ans;
    }
};
```



## 21[腐烂的橘子](https://leetcode.cn/problems/rotting-oranges/description/?envType=study-plan-v2&envId=top-100-liked)

```c++
// bfs 按层进行搜索，利用queue进行，每一次遍历len(queue.size)的n个元素再将其子节点插入队尾，直至队列元素为空
// 注意点：由于是网格搜索 需要四个方向向量，由于需要记录下横纵坐标 所以需要pair<int,int>
// pair插入 {} 以及.first .second
// 适当时候可以添加visited
// 此题按照腐烂的橘子一层层遍历搜索，刚开始从初始的烂橘子开始，再扩展到新添加的烂橘子这一层
class Solution {
public:
    int orangesRotting(vector<vector<int>>& grid) {
        int nr=grid.size();
        int nc=grid[0].size();
        int directions[4][2]={{-1,0},{1,0},{0,-1},{0,1}};
        queue<pair<int,int>> q;
        int freshcount=0; //记录新鲜橘子个数
        for(int i=0;i<nr;i++)
        {
            for(int j=0;j<nc;j++)
            {
                if(grid[i][j]==2)
                    q.push({i,j});
                else if(grid[i][j]==1)
                    freshcount++;
            }
        }
        if(freshcount==0) //没有新鲜橘子就0分钟腐烂
            return 0;
        if(q.empty()&&freshcount) //没有烂橘子有新鲜橘子就-1return
            return -1;
        int ans=0;
        while(!q.empty())
        {
            int n=q.size(); //每层数量
            for(int i=0;i<n;i++)
            {
                pair<int,int> p=q.front(); //取出对头元素
                q.pop();// 弹出
                for(auto it:directions)// 遍历四个方向等价为for(int *it:directions)
                {
                    int x=p.first+it[0];
                    int y=p.second+it[1];
                    if(x>=0&&x<nr&&y>=0&&y<nc&&grid[x][y]==1)
                    {
                        grid[x][y]=2;
                        q.push({x,y});
                        freshcount--;
                    }
                }
            }
            if(!q.empty()) //由于最后一层没有添加新橘子 所以ans不要多加1 
                ans++;
        }
        if(freshcount==0)
            return ans;
        else
            return -1;
    }
};
```

## p22[课程表](https://leetcode.cn/problems/course-schedule/description/?envType=study-plan-v2&envId=top-100-liked)

```c++
// 由于考虑先后关系像课程表、工程调度问题都需要拓扑排序进行处理
// 这里使用入度为0思想进行处理，即使用广度优先 和queue
// 总节点个数就是numCourses





class Solution {
public:
    // 使用广度优先遍历实现拓扑排序更加清晰
    // 广度优先采用的从入度这个思想即入度为0
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        vector<vector<int>> edges(numCourses); //相当于邻接表
        vector<int> indegree(numCourses,0); // 入度表
        for(int i=0;i<prerequisites.size();i++)
        {
            edges[prerequisites[i][1]].push_back(prerequisites[i][0]);// 后往前是边的方向
            indegree[prerequisites[i][0]]++;
        }
        queue<int> q;
        for(int i=0;i<indegree.size();i++) // 初始化第一步入度为0的节点
        {
            if(indegree[i]==0)
                q.push(i);
        }
        int count=0; //判断总个数是否为跟numCourses相同
        while(!q.empty())
        {
            int front=q.front();
            q.pop();
            count++;
            for(auto it:edges[front])
            {
                indegree[it]--;
                if(indegree[it]==0) //新的入度为0的点
                    q.push(it);
            }
        }
        return count==numCourses;
    }
};
```

