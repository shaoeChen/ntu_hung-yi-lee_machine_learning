# 李宏毅_ML_Lecture_4
###### `Hung-yi Lee` `NTU` `Machine Learning`
[課程撥放清單](https://www.youtube.com/channel/UC2ggjtuuWvxrHHHiaDH1dlQ/playlists)
## ML Lecture 4: Classification
[課程連結](https://www.youtube.com/watch?v=fZAZUYEeIMg&index=9&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49)  
### Classification
![](https://i.imgur.com/VIX7fUG.png)

* 是否借款
* 醫療診斷
* 手寫辨視
* 人臉辨視
### Example Application
![](https://i.imgur.com/tttXImn.png)

範例：輸入一隻寶可夢來預測它的屬性類別

註：寶可夢有18種屬性
### Example Application
![](https://i.imgur.com/RzBaPjD.png)

首先要確定特徵，將寶可夢的相關資訊數值化做為特徵。(如上圖條列)
### How to do Classification
![](https://i.imgur.com/x7WXmb3.png)

需求第一步，我們要收集訓練資料集，這邊也提到為什麼不能將分類問題以迴歸方式來處理。

舉例來說，當輸出接近1的時候就當做A類，接近-1的時候就當做B類
### Why not Regesssion
![](https://i.imgur.com/m0smPnr.png)

模型：$y=b+w_1x_1+w_2x_2$，藍色為類別1，紅色為類別2，雖然左圖來看分佈有順利的區隔，但如果出現離群值(如右圖)會造成整條迴歸線受離群值影響而造成迴歸線性切割無法順利區隔。
### Ideal Alternatives
![](https://i.imgur.com/E8dWUEY.png)

作法上我們可以這麼處理：    
* 在function內再加入一個function(g)來轉換
    * 當g(x)>0的時候就當做是類別1，否則即為類別2
* 損失函數單純統計錯誤次數?
    * 但是這樣做是無法微分的(無法梯度下降)
* 用其它的方法
    * svm、perceptron
### Two Boxes
![](https://i.imgur.com/sicQxJi.png)

從機率問題來談怎麼處理。

兩個盒子，裡面各有藍綠兩個色球，如果抽到一個藍色的球，那這球從box1抽的機率有多少?
### Two Classes
![](https://i.imgur.com/CGYTT8V.png)
 
將Box變為class，給一個x(要分類的對象)，它屬於某個class的機率為何?    
* $P(C_1)$:從class1抽出來的機率
* $P(C_2)$:從class2抽出來的機率
* $P(x|C_1)$:從class1抽出x的機率
* $P(x|C_2)$:從class2抽出x的機率

有了上面四種機率，就可以計算出$P(C_1|x)$的機率(x是class1的機率)，而這四種機率就是從我們的訓練資料去估測出來的，這種想法即為Generative Model，可以計算某一個x出現的機率    

$P(x)=P(x|C_1)P(C_1)+P(x|C_2)P(C_2)$    
### Prior
![](https://i.imgur.com/RzNK4wd.png)

* Class1是水系寶可夢
    * 數量：79
* Class2是一般系寶可夢
    * 數量：61

從兩類別中取得一隻寶可夢的機率如下：    
$P(C_1)=\dfrac{79}{79+61}=0.56$
$P(C_2)=\dfrac{61}{79+61}=0.44$
### Probability from Class
![](https://i.imgur.com/iRfhpea.png)

我們從水系神奇寶貝中撈到海龜的機率如何計算?$P(x|C_1)=?$

註：水系數量79
### Probability from Class - Feature
![](https://i.imgur.com/wQuos90.png)

每一隻寶可夢(神奇寶貝)都帶有它的所屬特徵(feature)，這特徵述敘是個向量(vectory)，將其中兩個特徵取出可視化觀察。

圖上每一個點都代表一隻寶可夢，現在有一個不存在這79隻寶可夢資料的新資料，它是屬於海龜的機率有多少?(不會是0)
### Gaussian Distribution
![](https://i.imgur.com/mlkKCAA.png)
![](https://i.imgur.com/In29csy.png)

高斯分佈，它的輸入是一個向量x，輸出是x被抽取到的機率(不全然是機率)，這個機率的分佈是由$\mu$(mean)與$\Sigma$(covariance)    
* $\mu$:vectory
* $\Sigma$:matrix

1. 不同的$\mu$相同的$\Sigma$，機率分佈最高點不一樣
2. 相同的$\mu$不同的$\Sigma$，機率分佈最高點一樣，但是分散的程度不一樣

註：Gaussian Distribution:高斯分佈    
註：covariance:共變異數
### Probability from Class
![](https://i.imgur.com/vc0qR6p.png)

我們從Gaussian中取出79個點(就是資料集中的79個水系神奇寶貝)，現在給我們一個新的點(不存在79個資料集中的新資料)，就可以利用Gaussian Distribution function來計算出抽到該點的機率。

以分佈來看，該點愈接近中心點被抽到的機率會愈高，離中心愈遠被抽到的機率則愈低。

所以，如何找出$\mu$與$\Sigma$就是一個問題。
### Maximum Likelihood
![](https://i.imgur.com/w7cle4N.png)

任何一個Gaussian都有可能找出這79個點，只是它的可能性(Likelihood)是不相同的，並且每一個點的機率都不會是0，以右上的分佈為例，離它最遠的點機率很低，但卻不會是0。

上圖兩個分佈為例，左邊Gaussian找出這79個點的機率會較右邊Gaussian來的高，只要有$\mu$與$\Sigma$我們就可以計算出這個Gaussian的Likelihood    

也就是這個Gaussian抽到這79個點的機率連乘積    
$L(\mu,\Sigma)=f_{\mu,\Sigma}(x^1),f_{\mu,\Sigma}(x^2)...f_{\mu,\Sigma}(x^{79})$

註：L非指成本函數
### Maximum Likelihood
![](https://i.imgur.com/EUETwJx.png)

所以我們現在要找出一個Gaussian($\mu^*, \Sigma^*$)，這個Gaussian是找出這79點的Lieklihood是最大的。

* $\mu^*=\dfrac{1}{79}\sum_{n=1}^{79}x^n$
    * 將79個x vectory加總之後取平均
* $\Sigma^*=\dfrac{1}{79}\sum_{n=1}^{79}(x^n-\mu^*)(x^n-\mu^*)^T$
### Maximum Likelihood
![](https://i.imgur.com/M8gdJ4w.png)
![](https://i.imgur.com/T0Sxwxt.png)

套入公式來計算，求出兩個calss的各別$\mu$與$\Sigma$，也就可以計算出$P(C_1|X)$
### How's the results?
![](https://i.imgur.com/RDxfLUU.png)

上圖左是水系(藍點)與一般系(紅點)在兩個特徵上的分佈，每一個點都可以計算它是$C_1$的機率。    
上圖右很清楚的看的出來，它們之間並沒有一個很明顯的決策邊界，測試資料集也僅47%的正確率，即使用了七個特徵，最後得到的結果還是只有54%的正確率。
### Modifying Model
![](https://i.imgur.com/9C12cI6.png)

一般來說，比較少會兩個類別即就分兩個$\Sigma$(covariance)，而是共用，以此降低參數，也降低overfitting的機會。
### Modifying Model
![](https://i.imgur.com/XQRuBeQ.png)

調整式子，讓水系與一般系神奇寶貝擁有相同的covariance，計算它們的likelihood。    
$L(\mu^1, \mu^2, \Sigma)=f_{u^1,\Sigma}(x^1)f_{u^1,\Sigma}(x^2)...f_{u^1,\Sigma}(x^79)f_{u^2,\Sigma}(x^80)f_{u^2,\Sigma}(x^81)...f_{u^2,\Sigma}(x^140)$

$\mu^2,\mu^2$不變，而$\Sigma$則依各自類別總數做加權平均$\dfrac{79}{140}\Sigma^1+\dfrac{61}{140}\Sigma^2$
### Modifying Model
![](https://i.imgur.com/6RPsH79.png)

調整共用covariance之後，決策邊界變為線性模型，並且考慮所有特徵之後正確率提升為73%。
### Three Steps
![](https://i.imgur.com/4AkgvHo.png)

機率模型三步驟：
* Model
    * 有$P(C_1),P(C_2),P(x|C_1),P(x|C_2)$
    * 不同的資料分佈就有不同的function
        * 不同的$\mu$(mean)與$\Sigma(covariance)$
    * $P(C_1)$>0.5, class=1
* Goodness of a function
    * 找出最大化產生資料集的likelihood
* Find the best function
### Probability Distribution
![](https://i.imgur.com/hr1tuZ5.png)

不一定要使用高斯分佈，也可以使用其它的，簡單的機率模型，參數少就high bias，low variance。    

x是一個1維向量，假設每一個特徵分佈是獨立的，則$P(x|C_1)=P(x_1|C_1)P(x_2|C_1)....P(x_K|C_1)$(**Naive Bayes Classifier**)，但以這種方式去建這次的案例得到的結果是不好的，可能是過於簡單，或許特徵間還是有相關性存在。

註：binary特徵可能會以bernoulli distribution來假設(柏努力)    
### Posterior Probability
![](https://i.imgur.com/JjeVMnM.png)

* $P(C_1|x)=\dfrac{P(x|C_1)P(C_1)}{P(x|C_1)P(C_1)+P(x|C_2)P(C_2)}$
    * 上下同除分子
    * $\dfrac{1}{1+\frac{P(x|C_2)P(C_2)}{P(x|C_1)P(C_1)}}$
        * 分母$\frac{P(x|C_2)P(C_2)}{P(x|C_1)P(C_1)}$取對數
        * $z=ln\frac{P(x|C_1)P(C_1)}{P(x|C_2)P(C_2)}$
            * $P(C_1|x)=\dfrac{1}{1+exp(-z)}=\sigma(z)$
            * sigmoid function
### Posterior Probability
![](https://i.imgur.com/EeRL1AK.png)
![](https://i.imgur.com/Zd8X8R0.png)
![](https://i.imgur.com/Si43g8h.png)
![](https://i.imgur.com/nHGztv3.png)

$P(C_1|x)=\sigma(z)$    

$z=ln\dfrac{P(x|C_1)P(C_1)}{P(x|C_2)P(C_2)}=ln\dfrac{P(x|C_1)}{P(x|C_2)}+ln\dfrac{C_1}{C_2}$    

最後推導出公式如上圖    
$P(C_1|x)=\sigma(w \cdot x+b)$，但最後發現，我們需要的是w與b，那為何需要計算這麼多的機率再來求得w與b，如果可以直接計算出w與b那就可以直接求解了。    

註：$N_1, N_2$:類別1,2的資料集數量    
註：$P(x|C_1),P(x|C_2)$:高斯機率分佈    
註：之前討論過，共用covariance，因此$\Sigma_1=\Sigma_2=\Sigma$


