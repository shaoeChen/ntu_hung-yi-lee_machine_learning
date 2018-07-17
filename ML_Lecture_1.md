# 李宏毅_ML_Lecture_1
###### `Hung-yi Lee` `NTU` `Machine Learning`
[課程撥放清單](https://www.youtube.com/channel/UC2ggjtuuWvxrHHHiaDH1dlQ/playlists)
## ML Lecture 1: Regression - Case Study
[課程連結](https://www.youtube.com/watch?v=fegAeph9UaA&index=3&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49)
### Regression
![](https://i.imgur.com/BjXziAV.png)

迴歸可以做的事：
* PM2.5
* 股票價格
* 自駕車
    * input：各種sensor資訊
    * ouput：方向盤角度
* 推薦系統
    * 購買的可能性
### Example Application
![](https://i.imgur.com/Bl3Inh6.png)

寶可夢的CP值(戰鬥力)!!     
找一個function來預測寶可夢的CP值，太低就省下資源去進化強的寶可夢了。    

符號約定：    
* 以下標來表示輸入的某個component(特徵)
    * $X_{cp}$:代表這隻妙娃種子的cp
### Step_1:Model
![](https://i.imgur.com/jiqMWNr.png)


第一步：找一個function set(Model)：
* $y=b+w*x_{cp}$
    * 線性模型
    * $y=b+\sum w_ix_i$
        * $x_i$:feature
        * $w_i$:weight
        * $b$:bias

$w、b$:參數，可以是任何數值
$x_{cp}$:進化前的cp
$y$:進化後的cp
### Step_2:Goodness of Function
![](https://i.imgur.com/1PU4wMH.png)


收集Training Data，這是監督式學習的案例，所以我們會收集輸入與輸出(此例迴歸，故輸出為數值)。

符號約定：    
* 以上標來表示某個object的編號(輸入資料第i筆)
    * $x^1,x^2....x^n$
    * 以$\hat{y}^1$來表示第i筆資料的輸出

### Step_2:Goodness of Function
![](https://i.imgur.com/78ukqjK.png)


假設我們抓了十隻寶可夢，將它的進化前化cp繪製出如上圖。
### Step_2:Goodness of Function
![](https://i.imgur.com/Bmslbg4.png)


有了資料之後就可以定義function好壞，這個用來定義的function稱為Loss function(L)
$L(f)=L(w,b)=\sum_{n=1}^{10}(\hat{y}^n-(b+w*x^n_{cp}))^2$

直觀來看，當估測誤差愈大就代表這個function愈不好
### Step_2:Goodness of Function
![](https://i.imgur.com/HbKOcO1.png)

將參數$w,b$可視化繪出，愈藍代表Loss function愈小，愈紅代表愈大
### Step_3:Best Function
![](https://i.imgur.com/7DOlVvT.png)

定義好Loss function之後，就可以從function set(Model)中找到一個最好的function，也就是估測誤差最小的那一個，而找到這個最佳解的方式就是Gradient Descent。    
Gradient Descent並非只能應用於解線性迴歸，只要可微分都適用。

註：修過線性代數的應該會知道closed-form solution
###  Step_3:Gradient Descent
![](https://i.imgur.com/gWNBfcW.png)

**用一個簡單的假設，僅一個參數情況下**    
1. 隨機初始化$w$
2. 計算$w$對L的微分(切線斜率)
    * 正斜率：減少w
    * 負斜率：增加w
3. $w = w-\eta\frac{dL}{dW}$
4. 重覆2,3步驟

註：梯度下降的大小取決定梯度以及學習效率(learning rate)
註：Gradient Descent後續會有更詳細說明的課程
註：linear regression不存在local optimal
###  Step_3:Gradient Descent
![](https://i.imgur.com/5mFGt7B.png)

**兩個參數**    
1. 隨機初始化$w,b$
2. 計算$w$對L與$b$對L的微分(切線斜率)
    * 正斜率：減少w,b
    * 負斜率：增加w,b
3. 
    * $w = w-\eta\frac{dL}{dW}$
    * $b = b-\eta\frac{dL}{db}$
4. 重覆2,3步驟
###  Step_3:Gradient Descent
![](https://i.imgur.com/RtTDN4k.png)

將成本函數可視化，這是一個由w、b決定的function，愈往中心代表Loss愈小，每次的迭代都會愈往中心移動。
### Step_3:Gradient Descent
![](https://i.imgur.com/KqM2Mod.png)

如果成本函數(Loss function)長的像上圖，那就可能會因為不同的初始化有不同的結果，可能是區域最佳，也可能是全域最佳。    
在Linear regression中，Loss function是convax(不存在區域最佳)，如右小圖，因此隨便一個初始化都會求得相同最佳解。    
### Step_3:Gradient Descent
![](https://i.imgur.com/1eWeptH.png)

這邊是關於偏微分的數學式：
* $L(w,b)=\sum^{10}_{n-1}(\hat{y}^n-(b+w*x^n_{cp}))^2$
* $\frac{\partial L}{\partial w}=\sum^{10}_{n=1}2(\hat{y}^n-(b+w*x^n_{cp}))(-x^n_{cp})$
* $\frac{\partial L}{\partial b}=\sum^{10}_{n=1}2(\hat{y}^n-(b+w*x^n_{cp}))(-b)$

### How's the results?
![](https://i.imgur.com/wn0y7dr.png)

最後求得解為如上圖呈現，透過計算點與線之間的距離(垂直x軸)就可以計算出均方誤差，但是我們真正想關心的是未知的資料。
### How's the results?-Generalization
![](https://i.imgur.com/glQsBuD.png)

我們另外準備了10隻寶可夢的資料(testing data)要來驗證模型的擬合是否完善，求得的解似乎比剛才的training data還要大一些。    
觀察模型不難發現，在cp兩極值的部份對預測值來說是較為不準的，這代表我們可能需要一個更複雜的Model。    
### Selecting another Model
![](https://i.imgur.com/AVA1vk3.png)

將模型調整為二次式$y=b+w_1*x_{cp}+w_2*(x_{cp})^2$，以此求得最佳解再重新可視化，模型效能也有更好的回饋。
### Selecting another Model
![](https://i.imgur.com/EDvnrl2.png)
![](https://i.imgur.com/FvLZjKY.png)
![](https://i.imgur.com/shsdxpa.png)

引入更複雜的模型，再改以三次方、四次方....但結果似乎是愈來愈糟。
### Model Selection
![](https://i.imgur.com/xl8HcSC.png)

直觀來看，五次方的function space包含著四次方，這是因為將五次方項目設置為0的時候等於四次方，四次方則包含著三次方(如上圖呈現)，因此愈複雜的Model就包含愈多的function，也就更有可能找到一個function讓error rate愈來愈低。
### Model Selection
![](https://i.imgur.com/NL4aC1c.png)

但是在testing data卻不是這麼一回事，在四次式之後有著反效果，這種情況稱為**overfitting**，並非愈複雜的Model就會有更好的效果。
### Let's collect more data
![](https://i.imgur.com/QiKkRcx.png)

在取得更多寶可夢資料之後，似乎這中間另外有一個力量在引導進化後的變化，這個部份即是物種。
### What are the hidden factors?
![](https://i.imgur.com/7D4qoV4.png)

將不同的物種以不同的顏色來標註可以明顯的看出，單純考量cp是不足以預測進化後的cp，因為物種的影響也很大。
### Back to step 1: Redesign the Model
![](https://i.imgur.com/dRgtwZA.png)
![](https://i.imgur.com/8KEc68t.png)
![](https://i.imgur.com/fIW5xvu.png)

針對不同的物種給了不同的function，當寶可夢是比比鳥系列的時候，就只有比比鳥相關的function會是1，其餘皆為0，也就是說，物種也是我們這個function中的一個feature。    
最後我們就會得到適應各種不同物種的線性函數，可以更擬合資料集，伊布的差異可能沒救，那因為他會變化成不同屬性的精靈，其它的差異可能來自於**random**(也許在進化的時候cp加了一個random參數)
### Are there any other hidden factors?
![](https://i.imgur.com/FdORg1U.png)

也許跟身高、體重或是hp會有關聯?這需要**domain knowledge**來協助。
### Back to step 1: Redesign the Model
![](https://i.imgur.com/QgnTEYT.png)

將所有參數混合，但結果不如預期，testing error太過，明顯的overfitting
### Back to step 2: Regularization
![](https://i.imgur.com/TaBIzjm.png)

原始的Loss function只考慮誤差平方的加總，加入另一個正規項(Regularization)如下：    
* $L=\sum_n(\hat{y}^n-(b+\sum w_ix_i))^2+\lambda\sum(w_i)^2$
    * $\lambda$:常數

參數$w_i$接近0的function是較為平滑的，意思是當輸入有變化的時候，輸出對這變化是不敏感的，因為$w_i$是接近0的，乘上$x_i$的值也是小的，也因此當資料有噪音的時候它的影響也會是小的。

### Regularization
![](https://i.imgur.com/6HfZCk4.png)

透過不同的$\lambda$觀察不同的模型結果：
* $\lambda$愈大
    * 代表考慮smooth的那個Regularization影響力愈大
    * 訓練資料誤差變大
        * 因為我們傾向考慮參數值$w_i$，減少考慮error
    * 測試資料誤差飄動
        * 太平滑反而又得到糟糕的結果

正規項通常不考慮偏差單元(bias)，調整function的平滑度(smooth)跟bias是沒有關係的(它只影響上下)。
### Conclusion & Following Lectures
![](https://i.imgur.com/BG6W6dg.png)

總結：
1. 了解影響寶可夢進化後的cp有目前cp與物種
2. 簡單談了Gradient descent(梯度下降)
3. 說明了Overfitting與Regularization(正規化)
4. 最後如何確認我們的模型是好的?