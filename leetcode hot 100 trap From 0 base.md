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



