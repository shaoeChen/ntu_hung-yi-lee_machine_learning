# 李宏毅_ML_Lecture_2
###### `Hung-yi Lee` `NTU` `Machine Learning`
[課程撥放清單](https://www.youtube.com/channel/UC2ggjtuuWvxrHHHiaDH1dlQ/playlists)
## ML Lecture 2: Where does the error come from?
[課程連結](https://www.youtube.com/watch?v=D_S6y0Jm6dQ&index=5&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49)
### Review
![](https://i.imgur.com/eOonkMa.png)

回顧上節課的結論，愈複雜的Model不一定會有愈好的結果。    
這次要討論的是error的來源：
* bias
* variance
### Estimator
![](https://i.imgur.com/KnmAM9J.png)

上節課建立的模型，$\hat{f}$輸入寶可夢資料，得到輸出(進化後的cp)$\hat{y}$，但是我們並不會知道那個真正的$\hat{f}$，透過訓練資料集我們只能得到可能的$f^*$

而$f^*$與$\hat{f}$的差異即來自於bias與variance

註：$f^*$是$\hat{f}$的估測值
### Bias and Variance of Estimator
![](https://i.imgur.com/LEg8078.png)

從統計與機率的概念來了解，我們有一個『variable x』，我們希望估測它的mean，作法如下：
1. 取sample N個點來計算平均值m
    * $m \neq \mu$
2. 如上圖右，不同的實驗得到了不同的m
    * 很難會有$m=\mu$
3. 計算期望值
    * 所有m的期望值會等於$\mu$

每一個m它可能不會相等於$\mu$，但是它們的期望值會正好等於$\mu$，好比你打靶的時候雖然瞄準中心，但是因為諸多因素而導致最後散佈在中心的週圍。

註：mean假設為$\mu$
註：variance假設為$\sigma^2$
### Bias and Variance of Estimator
![](https://i.imgur.com/Y3csd9R.png)

散佈的範圍取決於variance，而variance取決於你取多少的sample，sample多則分佈較為集中，反之則分散。

註：**簡報上的Larger與Smaller反了**
註：$Var[m]=\frac{\sigma^2}{N}$
註：mean假設為$\mu$
註：variance假設為$\sigma^2$
### Bias and Variance of Estimator
![](https://i.imgur.com/sj0T9Q3.png)

計算出m之後，以此估算出$s^2$，這是$\sigma^2$的估測，如$\mu$一般，它會散佈在$\sigma^2$的週圍。    
計算$s^2$的期望值，它並不會等於$\sigma^2$，普遍比$\sigma^2$還要小($\frac{N-1}{N}\sigma^2$)。    
一樣的，當N變大，兩者之間的估測差距就會變小(上圖右)

註：$Var[m]=\frac{\sigma^2}{N}$
註：mean假設為$\mu$
註：variance假設為$\sigma^2$
### Bias and Variance of Estimator 
![](https://i.imgur.com/2SieaYW.png)

* 目標：$\hat{f}$
* 預測：$f^*$
* $f^*$的期望值(平均)：$\bar{f}$

分類器的bias為$\bar{f}$，就是一開始就沒有瞄到紅心，而我們期望會打到$\bar{f}$卻落在$f^*$，這中間的差異即為variance，這兩個部份即為誤差來源。

### Parallel Universes
![](https://i.imgur.com/8Vq9mDP.png)

模型的訓練雖然只會有一次即得結果，但是相同的模型可以以不同的資料來做訓練以取得不同的$f^*$，舉課程例來說，第一次訓練的可能是用編號1-10的寶可夢，第二次訓練用編號11-20，相同的模型不同的數據資料。

### Parallel Universes
![](https://i.imgur.com/nzrmD2r.png)

假設做了100次不同的模型訓練(這100次皆不同的寶可夢)，並將模型可視化，結果如上圖，會發現什麼都有可能會發生。
### Variance
![](https://i.imgur.com/oYCN0sV.png)

結果來看，選擇較為簡單的模型其Variance較低(較為集中)，這是因為愈簡單的Model，資料受其影響性愈小，以最極端安例來看，y=c，那不管怎麼選擇都是c，。    
 
### Bias
![](https://i.imgur.com/mSXr3L9.png)

$E[f^*]=\bar{f}$，如果這個均值接近$\hat{f}$，那就是Low Bias，但是我們並不會知道$\hat{f}$，所以要假設，假設一個$\hat{f}$
### Bias
![](https://i.imgur.com/a1Z8Q9X.png)

* 黑線：假設的$\hat{f}$
* 紅線：5000次的模型訓練
* 藍線：5000次的訓練平均

二次式來看，藍黑差異不小，但以三次式來看雖然每次的差異很多(紅線)，但是平均來看卻是很接近(藍線)
### Bias
![](https://i.imgur.com/3Q5hxM9.png)

結果來看，比較簡單的模型會有比較大的Bias，比較複雜的模型則會有比較小的Bias。

模型是一個function set，也就是範圍，在定義一個模型的同時也產生了一個function set，也定義住範圍，簡單的模型則相對的所定義的範圍較小，而同時的這個範圍可能是不包含目標(target)，這時候不管怎麼處理都是high bias。
### Bias vs Variance
![](https://i.imgur.com/u4Xgw5z.png)

上圖說明著，簡單的模型會有著Low Variance/High Bias的特性(Overfitting)，隨著模型的複雜，會逐漸變成High Variance/Low Bias(Underfitting)。
### What to do with large bias?
![](https://i.imgur.com/7KYLbNk.png)

當模型無法擬合訓練資料集，那就是High Bias(Underfitting)，如果訓練資料集狀況良好，但測試資料集狀況不佳，那就是High Variance(Overfitting)

* High Bias
    * 加入更多的特徵
    * 更為複雜的模型
### What to do with large variance? 
![](https://i.imgur.com/aXw9SjA.png)

* High Variance
    * 收集更多資料
        * 資料增強
    * 正規化
        * 可能影響bias
### Model Selection
![](https://i.imgur.com/T3nj7dL.png)

在Bias與Variance中找尋一個平衡點，選擇一個適合的模型，但需注意到，我們手上的測試資料集始終是一個已知的資料，到實際的未知資料中不見得會有相等的好結果。
### Cross Validation
![](https://i.imgur.com/KpdJHBs.png)

作法上，可以將訓練資料集再拆分為訓練集與驗證資料集
### N-fold Cross Validation
![](https://i.imgur.com/R3GyF0x.png)

如果對於單一的驗證資料集不信任的話，可以透過多折的方式來驗證模型效能，再計算平均誤差，以此方式選擇模型。
