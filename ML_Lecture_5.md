# 李宏毅_ML_Lecture_5
###### `Hung-yi Lee` `NTU` `Machine Learning`
[課程撥放清單](https://www.youtube.com/channel/UC2ggjtuuWvxrHHHiaDH1dlQ/playlists)
## ML Lecture 5: Logistic Regression
[課程連結](https://www.youtube.com/watch?v=hSXFuypLukA&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=10)  
### Step 1: Function Set
![](https://i.imgur.com/wknDwRZ.png)    
![](https://i.imgur.com/7AXZUhG.png)    

上節課最後提到，當機率$P_{w,b}(C_1|x) \geq 0.5$則輸出$C_1$，否則即輸出$C_2$    
1. $P_{w,b}(C_1|x)=\sigma(z)$
2. $\sigma(z)=\dfrac{1}{1+exp^{(-z)}}$
3. $z=w\cdot x+b=\sum_iw_ix_i+b$

而Function set即為所有w、b的可能集合，給定一個資料集$x$是屬$C_1$的機率，這個模型即為Logistic Regression。    

### Step 2: Goodness of a Function
![](https://i.imgur.com/lyGsaZy.png)    

假設訓練資料集是依據Posterior Probability所產生的，所以給了一組的w、b就相對的決定了Posterior Probability，就可以計算這組w、b產生該訓練資料集的機率。    

其中可以最大化機率的參數就設置為$w^*$與$b^*$    

註：$L(w, b)$所指非成本函數，而是Likelihood    
### Step 2: Goodness of a Function
![](https://i.imgur.com/1iHFr9q.png)    
![](https://i.imgur.com/YL2dFJH.png)    

調整數學式，將原本要$maxL(w,b)$變更為$min-lnL(w,b)$以簡化計算(這是等價的)，再以1、0來表示$C_1,C_2$。    

整體所求的部份即為$\sum-[\hat{y}^nlnf_{w,b}(x^n)+(1-\hat{y}^n)ln(10f_{w,b}(x^n))]$，而這個部份即是Cross entropy(交叉熵)    

$Cross entropy:H(p,q)=-\sum_xp(x)ln(q(x))$，代表p、q有多接近，如果兩個分佈一樣的話，那所得即為0    

### Defferent between Logistic and Linear
![](https://i.imgur.com/3pXDYuc.png)    

上表是Linear Regression與Logistic Regression目前的差異，除了hypothesis不同之外，成本函數也是不同，那為何Logistic不能以Linear的成本函數求解即可?(後續說明)
### Step 3: Find the best function
![](https://i.imgur.com/Fe7qdAd.png)    
![](https://i.imgur.com/fKnwDD4.png)    

現在做的就是利用梯度下降(Gradient Descent)來最小化交叉熵(Cross Entropy)，作法上即是計算$-lnL(w,b)$對各$w_i$的偏微分。    

* $\dfrac{\partial lnf_{w,b}(x)}{\partial w_i}=\dfrac{\partial lnf_{w,b}(x)}{\partial z}\dfrac{\partial z}{\partial w_i}$    
    * $\dfrac{\partial z}{\partial w_i}=x_i$
    * $\dfrac{\partial lnf_{w,b}(x)}{\partial z}=\dfrac{\partial ln\sigma(z)}{\partial z}=\dfrac{1}{\sigma(z)}\sigma(z)(1-\sigma(z))$
        * 以圖示來看，在兩邊對$\sigma (z)$做偏微分的時候斜率是非常小的，中間的部份斜率最大。
* $\dfrac{\partial lnf_{w,b}(x)}{\partial w_i}=(1-\sigma(z)\cdot x_i)=(1-f_{w,b}(x^n))\cdot x_i^n$

另一部的推導過程也是相同，最後所得結果為$f_{w,b}(x^n)x^n_i$
### Step 3: Find the best function
![](https://i.imgur.com/RMWoXNx.png)    

最後將上面推導的式子帶入之後所得的結果是非常直觀的。    
$\dfrac{-lnL(w,b)}{\partial W_i}=\sum_n-(\hat{y}^n-f_{w,b}(x^n))x_i^n$    

權重的更新取決於三件事：    
1. Learning Rate($\eta$)(自己設置)
2. $x_i$(來自資料)
3. $\hat{y}^n-f_{w,b}(x^n)$(代表目前的預測與實際的差距)
### Defferent between Logistic and Linear
![](https://i.imgur.com/6YM4DGs.png)    

最後可以發現，Linear Regression與Logistic Regression在更新的部份也是執行一樣的步驟，不同的部份在於Logistic的$\hat{y}$是0或1，而Linear的$\hat{y}$是任意實數。
### Why not Square Error
![](https://i.imgur.com/GDyxxol.png)    
![](https://i.imgur.com/0KObdNF.png)    

如果以Linear Regression的成本函數來求Logistic的話，在$\hat{y}^n=1$的時候，不論$f_{w,b}(x^n)$是0或1所得的偏微分都是0，相反也是一樣的情況。    

### Cross Entropy vs Square Error
![](https://i.imgur.com/tBjYekg.png)    

圖示來看，以Square Error做成本函數會造成它的斜率非常小，可能一開始的時候就馬上卡住，加大學習效率也可能造成脫離最佳解的後果，但以Cross Entropy不會有此問題。
### Discriminative vs Generative
![](https://i.imgur.com/QSnMAUt.png)    
![](https://i.imgur.com/dpprWLG.png)    

不論是利用Logistic或是在之前機率所談的Posterior Probability，它們所使用的Model都是一樣的，但是它們所得的$w,b$是不相同的，原因在於『假設』上的不同。    
* Logistic
    * 找出$w, b$
    * 沒有任何假設
* Posterior Probability
    * 找出$\mu^1,\mu^2, \Sigma^{-1}$來計算$w, b$
    * 假設是高斯分佈(伯努力...)

相同的神奇寶貝問題，分別應用兩種方式來預測，所得是Discriminative有較佳的結果。    
### Generative vs Discriminative
![](https://i.imgur.com/h727NVJ.png)    

舉例說明，四筆資料，兩個特徵，第一筆兩個特徵皆為1，類別為1，其餘為類別2。    

給定一筆測試資料，兩個特徵皆為1，這時候直覺上它是類別1，但如果以Naive Bayes(樸素貝葉斯)所得結果就並非如此了。    

Navie Bayes：假設特徵之間是獨立的    
$P(x|C_i)=P(x_1|C_i)P(x_2|C_i)$    
### Generative vs Discriminative
![](https://i.imgur.com/WVENpJY.png)    

以Navie Bayes來計算，所得的類別是$C_1$的機率是小於0.5，因此應該是$C_2$，這個案例說明著，在Generative與Discriminative之間的差異。    

Generative有做了部份的假設，腦補，資料沒有說的但是假設它有。    

註：簡報$C_2$註記錯誤，寫成$C_1$    
### Generative vs Discriminative
![](https://i.imgur.com/Xde3Gfg.png)    

什麼時候適合Generative?    
1. 訓練資料集很少
    * 資料集多的時候Discriminative有較佳的擬合
    * Generative因為有假設的存在，所以資料集少的時候會有不錯的效果
2. 資料集中噪音多的時候
    * 因為假設與腦補反而可以忽視掉噪點
4. 以prior(先驗)與class-dependent probabilties(類別機率)可以應用於不同資料來源。
    * 如語音辨視 
        * 神經網路只是語音辨視的其中之一
### Multi-class Classification
![](https://i.imgur.com/9SFk6Jy.png)    
![](https://i.imgur.com/g3X53qy.png)    
 
在多類別的情況下會利用`softmax`來轉換機率，幾個重點：    
1. 轉出值一定是判於0、1之間
2. 總合一定是1
3. 負(z)的會變正(y)的

優化的對象即是輸出與目標之間的Cross Entropy    

註：取e之後，最大值與最小值的差距會拉大    
註：正確為$-\sum\hat{y}_ilny_i$    
### Limitation of Logistic Regression
![](https://i.imgur.com/QxybU3x.png)    
![](https://i.imgur.com/bTzbNco.png)    
![](https://i.imgur.com/lmUhqIW.png)    


logistic有很大的限制，在線性不可分的時候無法有效預測，以上例來看，不管你怎麼分割，始終無法讓機率是>0.5。    

如果堅持要使用lotistic的話，那就需要做特徵轉換，但是麻煩的是很難確定怎麼樣的特徵轉換可以滿足需求。    
### Limitation of Logistic Regression
![](https://i.imgur.com/OWzDu8N.png)    
![](https://i.imgur.com/cy81VAE.png)    
![](https://i.imgur.com/wO6KXHu.png)    

其中一種作法就是利用兩個logistic來做feature transformation，轉換之後再以另一個logistic來做預測，達到線性可分。    
### Neural Network
![](https://i.imgur.com/8HWREzm.png)    

把一個logistic的輸出當做另一個logistic的輸入，一個接一個，再給每個logistic一個很Powerful的名字『Neuron』，接起來就是Neural Network。    