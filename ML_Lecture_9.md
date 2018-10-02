# 李宏毅_ML_Lecture_9
###### `Hung-yi Lee` `NTU` `Machine Learning`
[課程撥放清單](https://www.youtube.com/channel/UC2ggjtuuWvxrHHHiaDH1dlQ/playlists)
## ML Lecture 9-1: Tips for Training DNN
[課程連結](https://www.youtube.com/watch?v=xki61j7z-30&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=16)      
### Recipe of Deep Learning
![](https://i.imgur.com/NueruTB.png)    

在完成模型訓練之後，首先確認模型在『訓練』資料集是否有好的結果，如果沒有就回頭確認前面的步驟有無可調整的部份，確認完訓練資料集就可以確認『測試』資料集，以此確認有無過適問題(Overfitting)。    
### Do not always blame Overfitting
![](https://i.imgur.com/mujBXjD.png)    

理論而言，56層的神經網路絕對擁有(或超過)跟20層相同模型效能，所以可能是模型參數的調校問題導致未能有此結果。    
### Recipe of Deep Learning
![](https://i.imgur.com/UrwNlkb.png)    

針對不同資料集(訓練、測試)效能不好有不同的處理方式，用了錯誤的方式調校除了無法提升效能之外還可能更糟。    
### Hard to get the power of Deep
![](https://i.imgur.com/6TNtY0A.png)    

以MNIST為例，在activation使用sigmoid的時候會發現到，愈深的神經網路並沒有得到愈好的結果，這並不是overfitting，這是highbias。    
### Vanishing Gradient Problem
![](https://i.imgur.com/Ev15GAe.png)    

這其中一個原因在於，當神經網路很深的時候，在input附近的幾個layer對loss的微分是很小的(梯度消失)，也因此該幾層的學習效率是很慢的，在它們還是幾乎亂數的同時，最後的輸出層已經差不多好了，但是它們的好是根據前面的亂數參數。    
### Vanishing Gradient Problem
![](https://i.imgur.com/UQIicec.png)    
![](https://i.imgur.com/7L6EWRw.png)    

直觀來看，偏微分就是某一個參數的小小變化對最後cost的影響有多少。    

第一個layer的某一個參數加上$\Delta w$，觀察它對最後的loss的影響。    

$\Delta w$很大的情況下，通過sigmoid之後會變小，因為sigmoid的特性是將極大與極小值壓縮到0與1之間，每通過一次就衰減一次，當神經網路愈深，它就會愈衰減。    

預防這問題的一個方法就是調整使用的啟動函數。    
### ReLU
![](https://i.imgur.com/Lj8BUvG.png)    

Relu:Rectified Linear Unit，這是目前較為常用的一個啟動函數。    

if z > 0: a = z    
if z <= 0: a = 0     

z: activateion's input    
a: activateion's output    

幾個理由使用Relu：    
1. 它快很多
2. 生物上的理由
3. 無窮多sigmoid疊加的結果就是relu
4. 可避免梯度消失問題
### ReLU
![](https://i.imgur.com/DwiJlKg.png)    
![](https://i.imgur.com/fPdxmhi.png)    


Relu有兩個作用域：
1. if z > 0: a = z    
    * 即為線性
3. if z <= 0: a = 0     

當neuron為零的時候它對神經網路是沒有影響的，將沒影響的neuron拿掉之後整個神經網路就是一個很瘦長的線性神經網路，再加上relu並不會壓縮輸出，因為大於零的情況下輸入等於輸出，就不會有梯度遞減的問題。    

但有一個問題，使用Relu之後整個神經網路看似變成線性神經網路，這跟我們希望它是非線性的初衷是不同的。    

實際上它還是一個非線性神經網路，在作用域會改變的情況下，它還是一個非線性神經網路。    

在微分問題中：    
1. if a=z: gradient=1
1. if a=0: gradient=0
### ReLU - variant
![](https://i.imgur.com/omyl5Lm.png)


Leaky Relu: 避免數值為負的時候偏微分為0，因此當z為負數時$a=0.01z$    
Parameteric Relu: $a=\alpha z$    
### Maxout
![](https://i.imgur.com/IT0NZEU.png)    

Maxout的概念類似於MaxPooling，不同之處在於MaxPooling是在Image上執行，Maxout是在Layer上執行。    
將neuron分群之後(事先決定)，乘上權重只輸出最大值，而幾個neuron為一個群組就是超參數的設置。    
### Maxout
![](https://i.imgur.com/kZS47no.png)

這邊說明的是Maxout如何做到Relu。    
上圖左是ReLU，藍線是它的線性函數，綠線是它的啟動函數，當值大於0的時候a=z。    
上圖右是Maxout，z1、z2是它的兩組權重，藍線是z1的線性，紅線是z2的線性，綠線是Maxout的輸出，可以發現，在w、b的設置與relu相同的情況下，Maxout的輸出是與Relu相同。    
### Maxout
![](https://i.imgur.com/a3IvM8X.png)    
![](https://i.imgur.com/6Cr2Up1.png)    

Maxout能透過學習來學習到更特別的啟動函數，每一個neuron根據不同的weight可以擁有不同的啟動函數。    
根據超參數(幾個neuron一個group)的設置它會有不同的啟動函數。    
### Maxout - Training
![](https://i.imgur.com/rWkbJYg.png)    

Maxout的訓練，假設紅框是群組最大值，即為output，它是一個線性(a=max(z))，將其餘neuron拿掉不看的時候，訓練的也是一個瘦長型的線性網路。    
對於沒被訓練到的neuron不需要特別的擔心，因為不同的輸入會有不同的啟動結果，這次沒有被訓練到的，也許下一次的輸入所對應的最大值是它，那就訓練的到了。    
### Adagrad Review
![](https://i.imgur.com/UvrNLtt.png)    

Adagrad主要給予各參數擁有不同的學習效率，以$\eta$除上過去梯度的平方和開根號(如上圖公式回憶)    
### RMSProp
![](https://i.imgur.com/AjDQtRa.png)    
![](https://i.imgur.com/EZvw0pu.png)    

但更多時候在處理機器學習的成本函數的時候使用Adagrad是不足的，必需要能更動態的調整學習效率    

* $w^1 \leftarrow w^0-\dfrac{\eta}{\sigma^0}g^0$
    * $\sigma^0=g^0$
* $w^2 \leftarrow w^1-\dfrac{\eta}{\sigma^1}g^1$
    * $\sigma^1=\sqrt{\alpha(\sigma^0)^2+(1-\alpha)(g^1)^2}$
* $w^3 \leftarrow w^2-\dfrac{\eta}{\sigma^2}g^2$
    * $\sigma^2=\sqrt{\alpha(\sigma^1)^2+(1-\alpha)(g^2)^2}$
* $\vdots$
* $w^{t+1} \leftarrow w^t-\dfrac{\eta}{\sigma^t}g^t$
    * $\sigma^t=\sqrt{\alpha(\sigma^{t-1})^2+(1-\alpha)(g^t)^2}$


與Adagrad不同的地方在於，原$\sigma$處單純的取$g^0,g^1...g^{t-1}$的平方和開根號，但在RMSProp多了超參數$\alpha$可調控。    
注意到，在RMSProp中，$\sigma^2$依然包含了$\sigma^1$與$\sigma^0$的值，應用超參數$\alpha$來調整過去與現在的權重。    


註1：RMSProp是由hinton在他的線上課程提出    
### Hard to find optimal network parameters
![](https://i.imgur.com/DXl8pys.png)    

訓練過程中，不僅會陷入local minima，也有可能進入saddle point或是plateau，但實務上並不需要過於擔心陷入local minima的問題，這機率比你所想的還要低。    
假設你陷入local minima的機率是P，有1000個參數，所有的參數都在local minima的機率就是$P^{1000}$，更何況你的神經網路所擁有的參數不可能只有1000。    
### In physical world
![](https://i.imgur.com/astXNDm.png)    

想像有一顆球從山頂滾下來，即使到了平坦的地方它還是不會停下來，因為它有慣性(不考慮阻力)，這個就叫做Momentum    

### Momentum
![](https://i.imgur.com/bi7t79h.png)    

Momentum的加入讓每次的梯度更新不再只有考慮梯度，而是現在這個時間點加上前一個時間點的移動方向。    

* init: $\theta^0$
    * Movement: $v^0=0$
        * 初始的時候梯度為0，故$v^0=0$
* 計算$\theta^0$的梯度
    * Movement: $v^1=\lambda v^0-\eta \nabla L(\theta^0)$
* 更新梯度：$\theta^1=\theta^0+v^1$
    * 如同加入慣性一般，不會只考慮該點上的梯度，而是加入考慮了上一次梯度
* 計算$\theta^1$的梯度
    * Movement: $v^2=\lambda v^1-\eta \nabla L(\theta^1)$
* 更新梯度：$\theta^2=\theta^1+v^2$
    * 綠色虛線為慣性，紅色實線為此次梯度所計算，兩者考慮之後產生一個新的藍色線方向


註：$\theta$上標所指為第n次的迭代更新    
註：Movementum中的$\lambda$為超參數    
### Momentum
![](https://i.imgur.com/7gB4cgD.png)    

$v^i$意指為過去所有梯度總合，只是說過去的影響力有多少，就是利用超參數$\lambda$來控制：    
* $v^0=0$    
* $v^1=0 - \eta \nabla L(\theta^0)$    
* $v^2=-\lambda \eta \nabla L(\theta^0) - \eta \nabla L(\theta^1)$    
* $\vdots$
### Momentum
![](https://i.imgur.com/WcUTkSG.png)    

比較直觀的來看Momentum，當梯度更新真的陷入了local的時候還是可以因為Momentum而繼續更新，並且在梯度更新中如果到了一個山坡段，只要Momentum夠力，就有機會越過山頂往更好的解過去。    
### Adam
![](https://i.imgur.com/mCGTq25.png)    

將RMSProp加上Momentum就可以得到Adam，目前實作上較多都直接採用Adam。    

### Early Stopping
![](https://i.imgur.com/lpItXoI.png)
   
參數調教正常情況下，訓練資料集的Loss會隨著迭次而逐漸降低，但測試資料集的資料分佈與訓練資料集並不全然相同，這種情況下是有可能在訓練資料集逐漸降低的同時而測試資料集是向上的，因此理想上我們會希望模型停在測試資料集的最低(如上圖所示Stop at here)    

註：這在keras上可以透過callback來達成    
### Regularization
![](https://i.imgur.com/kMUfnTs.png)
   
Regularization的作法，會在原始Loss function上再加上一個Regularization term(目前常見使用L2 norm)，並且注意到，執行Regularization是不考慮bias。    
加入Regularization的目的是讓function更加平滑(smooth)，而bias這跟事無關，因此不會考慮bias。    

$\theta=w_1,w_2,...$    
$L2_{norm}=||\theta||_2=(w_1)^2+(w_2)^2+...$    
### Regularization - Gradient
![](https://i.imgur.com/Et2Lixi.png)
   
加入正規項(Regularization)之後，梯度下降的式子如下：    
* $\dfrac{\partial L'}{\partial w}=\dfrac{\partial L}{\partial w}+\lambda w$
    * 參數更新：$w^{t+1}\rightarrow w^t-\eta\dfrac{\partial L'}{\partial w}=w^t-\eta(\dfrac{\partial L}{\partial w}+\lambda w^t)=(1-\eta\lambda)w^t-\eta\dfrac{\partial L}{\partial w}$

這代表每次在更新參數之前，我們都會將參數$w^t$乘上$1-\eta\lambda$，$\eta$，學習效率，會設置很小，$\lambda$，正規化超參數，也會設置很小，因此$1-\eta\lambda$是一個接近1的值，如0.999...，就是每次在更新參數之前都會乘上0.999..，這也造成了$(1-\eta\lambda)w^t$會接近0，但它並不會變0，因為還會再跟後面的項目$\eta\dfrac{\partial L}{\partial w}$做計算    

### Regularization -L1
![](https://i.imgur.com/sQAwmEy.png)
   
Regularization還有另一種稱為L1的正規化，不同於L2的部份在於$||\theta||$所計算是參數的絕對值總合，而不是平方和。    

* $L'(\theta)=L(\theta)+\lambda\dfrac{1}{2}||\theta||_1$
    * $\dfrac{\partial L'}{\partial w}=\dfrac{\partial L}{\partial w}+\lambda sgn(w)$
        * $sgn(w)$，如果$w$為正，就是1，為負就是-1

而加入L1正規化之後的參數更新，每次都會去減掉$\eta\lambda sgn(w^t)$，因此當$w$為正，就每次都減，會造成參數變小，$w$為負，就每次都加，會造成參數變大。    

比較L1與L2的差異：    
* 參數更新
    * L1減掉固定的值
    * L2乘上一個小於1固定的值
* 收斂速度
    * w愈大，於L2上更新的速度愈快
* 結果
    * L1參數有大有小
    * L2普遍參數都會接近於0

L2正規化之下，$w=1000000$，乘上0.99，它減掉的值是很大的，但是在L1正規化之後它所減的值是不變的，因此在L2正規化之下如果$w$很大，相對的也會收斂的快。    
### Dropout
![](https://i.imgur.com/IZMl0F4.png)    
![](https://i.imgur.com/intVnYt.png)    
![](https://i.imgur.com/K6OpK1s.png)    

訓練過程中，在更新參數之後對針對neuron做隨機的消除，如果一個neuron被選擇到要丟掉，那跟它相連的weight也會失去作用，這時候整個神經網路會變的細長。    

註：每次迭代所選擇要丟棄的neuron是隨機的。    
### Dropout
![](https://i.imgur.com/OkWb4qE.png)    

要注意到兩件事：    
1. 在測試的時候是不使用Dropout
2. 假設，dropout所設置捨棄的機率是p%，那testing的時候所有權重都要乘上(1-p)%
### Dropout - Intutive Reason
![](https://i.imgur.com/sEYHLkP.png)    
![](https://i.imgur.com/HJz4Q22.png)    
![](https://i.imgur.com/FP5gdr5.png)    


Dropout為何有用，直觀來看幾個理由：    
1. 就跟小李一樣，訓練的時候綁重物，戰鬥的時候解放。
2. 平常做論文的時候會發現有老鼠屎隊友，這時候就不得不出全力來處理論文，最後報告的時候發現，大家其實都很厲害。
3. 平常訓練的時候總是隨機丟棄了50%(P)的neuron，但在測試過程中並不會，這時候所計算得到的輸出會是訓練過程中的1/P倍，因此測試過程中將所有的權重都乘上P，讓輸出平等。

### Dropout is a kind of ensemble
![](https://i.imgur.com/C6sQvWx.png)    
![](https://i.imgur.com/azP7et8.png)    
![](https://i.imgur.com/Ecu2aMq.png)    

另類的直觀來看可以將Dropout視為一個ensemble，將很多個模型併成一個之後一起來決定結果(如同隨機森林)，其效果有如你在訓練$2^M$(最多)個netwroks，雖然每次只訓練一小小的minibatch，但是這些networks之間的參數是共享的。    
而測試的時候就是將資料丟進去那一大把的networks去做計算，每一個network都一個結果，最後平均結果回饋，但是這計算量很大，只是神奇的是，最終，你只要將所有的權重都乘上(1-P)，那所得的結果會接近那個均值。    
### Testing of Dropout
![](https://i.imgur.com/0bed9L6.png)    

舉例來看，兩個輸入做Dropout的可能有四種，其結果就是$\dfrac{1}{2}w_1x_1+\dfrac{1}{2}w_2x_2$，這跟你對權重直接乘上(1-P)是一樣的，前提是線性函數輸出，但事實是在非線性一樣得到很好的結果，因此，有人提出結論就是，在配合Relu的情況下Dropout所得到的結果是較佳的。    
## ML Lecture 9-2: Keras Demo2
[課程連結](https://www.youtube.com/watch?v=Ky1ku1miDow&index=17&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49)      
老師有測試各種參數的調校狀況，可以參閱。    

* 用了Dropout會讓訓練集的效能變差
* 使用adam的收斂較快
* 資料未做標準化會造成效能不佳
* 使用了gpu平行計算可以讓batch_size設大，但可能影響效能。

## ML Lecture 9-3: Fizz Buzz in Tensorflow (sequel)
[課程連結](https://www.youtube.com/watch?v=F1vek6ULo9w&index=18&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49)      