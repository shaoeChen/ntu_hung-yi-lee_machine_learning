# 李宏毅_ML_Lecture_3
###### `Hung-yi Lee` `NTU` `Machine Learning`
[課程撥放清單](https://www.youtube.com/channel/UC2ggjtuuWvxrHHHiaDH1dlQ/playlists)
## ML Lecture 3-1: Gradient Descent
[課程連結](https://www.youtube.com/watch?v=yKKNr-QKz2Q&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=6)
### Review: Gradient Descent
![](https://i.imgur.com/443sTei.png)
![](https://i.imgur.com/eneSHqI.png)

梯度下降，在機器學習的第三個步驟中，用來優化第二步驟(Cost Function)的演算法，每次的迭代更新利用計算參數對成本函數的偏微分乘上學習效率，一直到迭代結束。
$\theta = \theta-\eta\nabla L(\theta)$

註：投影片誤值max，應該為$\theta=argminL(\theta)$
### Learning Rate
![](https://i.imgur.com/gopI6d0.png)

學習效率的設置(左圖)：    
* 過小(藍線)
    * 需更多次的迭代才能到最佳解
* 過大(綠線)(黃線)
    * 可能永遠無法到達最佳解

每次的模型訓練都必需觀察其成本函數的走勢是否有順利的收斂(如右圖)

註：學習效率是模型中的超參數
### Adaptive Learning Rates
![](https://i.imgur.com/eQCqrap.png)

動態調整學習效率：    
* 隨著每次的迭代逐漸變小
    * 初始的時候成本函數較大，需要較大的學習效率來加速收斂
    * 幾次迭代之後，逐漸調降學習效率，讓它可以順利收斂在最低點
* 最好的方式是每一個不同的參數給它不同的學習效率
### Adagrad
![](https://i.imgur.com/6EoPgy4.png)

* $w^{t+1}=w^t-\eta^tg^t$    
    * $g^t$:偏微分的值
    * $\eta^t=\dfrac{\eta}{\sqrt{t+1}}$

我們將上面的標準梯度下降調整為Adagrad：    
* $w^{t+1}=w^t-\dfrac{\eta^t}{\sigma^t}g^t$
    * $\sigma^t$為過去所有微分項的平方平均數
        * 對每個參數而言值皆不同，如此即可造成每個參數不同的學習效率

註：t+1代表迭代次數
註：root mean square:平方平均數，亦稱方均根(參考[維基百科](https://zh.wikipedia.org/wiki/%E5%B9%B3%E6%96%B9%E5%B9%B3%E5%9D%87%E6%95%B0))
### Adagrad
![](https://i.imgur.com/0xs8qxB.png)

實作Adagrad：    
* $w^{1}=w^0-\dfrac{\eta^0}{\sigma^0}g^0$ 
    * $\sigma^0=\sqrt{(g^0)^2}$
* $w^{2}=w^1-\dfrac{\eta^1}{\sigma^1}g^1$ 
    * $\sigma^0=\sqrt{\dfrac{1}{2}[(g^0)^2+(g^1)^2]}$
* ....
* $w^{t+1}=w^t-\dfrac{\eta^t}{\sigma^t}g^t$ 
    * $\sigma^t=\sqrt{\dfrac{1}{t+1}\sum^t_{i=0}(g^i)^2}$
### Adagrad
![](https://i.imgur.com/ncptu4g.png)

數學式的部份，因為上下都有$\sqrt{t+1}$，因此可以消掉，整個式子即變為：    
* $w^{t+1}=w^t-\dfrac{\eta}{\sqrt{\sum^t_{i=0}(g^i)^2}}g^t$
### Contradiction?
![](https://i.imgur.com/jMXyu0X.png)

* Adagrad的參數更新會隨著迭代愈漸減緩

一般來說，梯度愈大則參數更新愈快，但使用Adagrad的時候，則會因為梯度愈大造成分母項也愈大，也因此參數更新是受限的。
### Intuitive Reason & Larger gradient, larger steps?
![](https://i.imgur.com/ZKwI9kl.png)
![](https://i.imgur.com/Ir2TOk9.png)


Adagrad所表示的是梯度的反差，以二次函數$y=ax^2+bx+c$對x微分為例，其結果如上圖下，它的最低點位於$-\frac{b}{2a}$。    
假設初始位於$x_0$，最好的方式就是一次到達最低點，兩點之間的距離為$\frac{|2ax_0+b|}{2a}$，而$|2ax_0+b|$即為$x_0$的微分項。    
當微分項愈大，代表離原點愈遠，如果跨出去的距離跟微分大小成正比，那可能就是最好的步伐，但這只限於一個參數情況。

### Comparison between different parameters
![](https://i.imgur.com/0VKd31d.png)

多參數情況下，上小節所說明的狀況卻不一定成立。    
左圖是成本函數的輪廓圖，c在$w_2$中的微分子相較a在$w_1$是較大(c的斜率較大)，但c離最低點的距離卻是比較近的。
### Second Derivative
![](https://i.imgur.com/wvCphil.png)

最佳步驟$\frac{|2ax_0+b|}{2a}$中的$2a$即是二次微分的結果$\dfrac{\partial^2y}{\partial x^2}=2a$，也就是二次微分如果比較大，那分母項就大，更新就會較小
### Comparison between different parameters
![](https://i.imgur.com/2DX5YQ9.png)

在把二次微分考慮進來之後改變如下：    
* 在$w_1$的方向上，二次微分項是小的(較平滑)
* 在$w_2$的方向上，二次微分項是大的

考慮二次微分項之後才能真正的反應現在所在位置與最低點的距離。
### Adagrad
![](https://i.imgur.com/N1CvXPo.png)

Adagrad：    
* $=w^{t+1}=w^t-\dfrac{\eta}{\sqrt{\sum^t_{i=0}(g^i)^2}}g^t$
    * $g^t$：代表一次微分項
    * $\sqrt{\sum^t_{i=0}(g^i)^2}$：代表二次微分項
        * 在不做真正的二次微分下以此方式做概估，減少計算成本
### Stochastic Gradient Descent
![](https://i.imgur.com/OkxlSEe.png)

$L=\sum(\hat{y}-(b+\sum w_ix^n_i))^2$    
Loss Function合理來說都會考慮所有資料集，再以所有資料集的總誤差來計算梯度下降，但Stochastic Gradient Descent(隨機梯度)只考慮一筆資料誤差，梯度也只考慮該筆資料。    
$L^n=\sum(\hat{y^n}-(b+\sum w_ix^n_i))^2$    
$\theta^i=\theta^{i-1}-\eta \nabla L^n(\theta^{i-1})$
### Stochastic Gradient Descent
![](https://i.imgur.com/PxYQVps.png)

隨機梯度下降與梯度下降的最大差異在於，梯度下降每次的迭代更新都會計算一次所有的資料誤差再做梯度下降，而隨機梯度下降則是每次的迭代都只計算一筆的誤差並且更新。    
也因此，在梯度下降做一次的迭代之後，可以隨機梯度已經看完全部而且更新完了。    
但是隨機梯度的收斂無法像梯度下降一樣很穩定的往最佳解前進，它的求解過程中較為震盪。
### Feature Scaling
![](https://i.imgur.com/dSJKDjp.png)

多特徵情況下，如果特徵間的分佈差異較大，可透過Feature Scaling來讓範圍在同一個級別。

### Feature Scaling
![](https://i.imgur.com/iyUBS6f.png)

在不做feature scaling的情況下，如果特徵間的差異過大，成本函數會呈現橢圓型(上圖左)，在經過縮放之後會呈現正圓(上圖右)。    
這影響著梯度下降的收斂，未經縮放的情況下需要更多次的迭代來收斂。

註：其中一個特徵的區間較大的話會明顯的影響模型。

### Feature Scaling
![](https://i.imgur.com/Q5b6Jwj.png)

縮放為均值為0，方差為1

### Question
![](https://i.imgur.com/ebwVmRN.png)

每次的迭代更新並不保證成本函數會下降
### Formal Derivation
![](https://i.imgur.com/EMAeaQK.png)

在成本函數的輪廓圖上給一個初始點，在範圍內找出一個最小值，再以該點為中心，以同樣的方式在範圍內找一個最小值前往。    
問題在如何找範圍內(紅圈)找一個讓成本函數最小的參數?
### Taylor Series
![](https://i.imgur.com/ibiHefN.png)

任一function(h(x))在$x=x_0$的這點是infinitely differentiable(無限微分?)，則$h(x)\approx h(x_0)+h'(x_0)(x-x_0)$
### E.g. Taylor Series
![](https://i.imgur.com/rgPEQBk.png)

上圖範例假設$h(x)=sin(x)$，$x_0=\dfrac{\pi}{4}$，所得數學式即如條列，將這項繪製的話即如圖示。    

單純考慮一次式的話，它的線與sin雖然非常不像，但是在$\dfrac{\pi}{4}$的地方卻是很像，其它的二次式以後的項目它的值都非常的小，我們可以忽略不看。

註：橙色線為sin的圖示，藍色線為各項次的趨勢線    
註：這是只有一個參數的情況
### Multivariable Taylor Series
![](https://i.imgur.com/fc0SFk3.png)

在多參數的狀況下依然可以以Taylor Seris來計算，在$h(x,y)$都很接近$h(x_0, y_0)$的情況下，平方項之後也是可以被忽略的。
### Back to Formal Derivation
![](https://i.imgur.com/9biNygT.png)

假設，在成本函數中一個中心點(a, b)並畫一個很小的圓圈，在這個圓圈內我們可以將Loss Function利用Taylor Series做簡化。    
* $L(\theta)\approx L(a, b)+\dfrac{\partial L(a,b)}{\partial\theta_1}(\theta_1-a)+\dfrac{\partial L(a,b)}{\partial\theta_2}(\theta_2-b)$   
    * $s=L(a,b)$    
    * $u=\dfrac{\partial L(a,b)}{\partial\theta_1}$
    * $v=\dfrac{\partial L(a,b)}{\partial\theta_2}$
* $L(\theta)\approx s+u(\theta-a)+v(\theta_2-b)$
### Back to Formal Derivation
![](https://i.imgur.com/DnILS2P.png)

簡化之後，s、u、v皆為常數項，我們要以此式來尋找最小化Loss function的$\theta_1,\theta_2$。    

$(\theta_1-a)^2+(\theta_2-b)^2 \leq d^2$

註：紅色圈的中心即為(a, b)
### Gradient descent - two variables
![](https://i.imgur.com/FOLhTJF.png)

$(\theta_1-a)^2+(\theta_2-b)^2 \leq d^2=\Delta\theta_1+\Delta\theta_2 \leq d^2$    

$L(\theta) \approx u\Delta\theta_1+v\Delta\theta_2$，將此式視為兩個向量，($\Delta\theta_1, \Delta\theta_2$)與($u, v$)，兩個向量做內積。問題回歸到如何選擇($\Delta\theta_1, \Delta\theta_2$)讓$L(\theta)$最小。

註：s忽略不看
### Gradient descent - two variables
![](https://i.imgur.com/q2EwZVF.png)

將($\Delta\theta_1, \Delta\theta_2$)轉到($u, v$)反方向，並拉長到紅圈的邊緣，這時候兩個向量做內積所得的值是最大的。

最大的負就是最小    
$\begin{bmatrix} \Delta\theta_1 \\ \Delta\theta_2 \end{bmatrix}=-\eta\begin{bmatrix} u \\ v \end{bmatrix}$    

$\begin{bmatrix} \theta_1 \\ \theta_2 \end{bmatrix}=\begin{bmatrix} a \\ b \end{bmatrix}-\eta \begin{bmatrix} u \\ v\end{bmatrix}$

註：a、b是中心點
### Back to Formal Derivation
![](https://i.imgur.com/tjl9EI2.png)

回頭將常數項u、v帶入我們剛才推出的式子，就是梯度下降了，而這個式子的成立就是建立在Taylor Series上，就是紅色的圈要夠小(學習效率要小)。

註：理論上要很小很小，但實務上只要小就可以
### More Limitation of Gradient Descent
![](https://i.imgur.com/KkI9gsf.png)

梯度下降的限制，很多時候它會陷入區域最佳，但更多時候你不知道它是不是在區域最佳。    
只要落入微分值是0的地方，它就會停止更新，但並非只有區域最佳會有這種情況，在鞍點(saddle point)的部份也會有相同的情況。
