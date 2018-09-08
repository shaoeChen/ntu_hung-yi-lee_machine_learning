# 李宏毅_ML_Lecture_7
###### `Hung-yi Lee` `NTU` `Machine Learning`
[課程撥放清單](https://www.youtube.com/channel/UC2ggjtuuWvxrHHHiaDH1dlQ/playlists)
## ML Lecture 7: Backpropagation
[課程連結](https://www.youtube.com/watch?v=ibJpTrp5mcE&index=12&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49)  
### Gradient Descent
![](https://i.imgur.com/vNiLrVP.png)

梯度下降，初始化參數($\theta^0$)先計算$\theta^0$對成本函數的偏微分，也就是計算所有權重w與偏差單元b對成本函數的偏微分，再利用這個結果來更新參數，一直迭代至結束。    
在神經網路的實作中因為參數太多了，所以利用backpropagation來處理，它並沒有不同於Gradient Descent，只是更有效率。
### Chain Rule
![](https://i.imgur.com/di2ZdhO.png)

backpropagation主要觀念是Chain Rule(鏈式法則)    

範例一：    
* y=g(x)    
* z=h(y)    

這時候x的變動會影響到y，而y的變動又影響到z。如果要求$\dfrac{dz}{dx}$(x對z的微分)的話，可以分成兩部份計算    

* $\dfrac{dz}{dy}\cdot\dfrac{dy}{dx}$

範例二：    
* x=g(s)
* y=h(s)
* z=k(x,y)

改變s的同時x與y也變了，而z也因此更著變化，因此，如果要計算$\dfrac{dz}{ds}$(s對z的微分)，可以這樣計算    
* $\dfrac{dz}{ds}=\dfrac{\partial z}{\partial x}\cdot\dfrac{dx}{ds}+\dfrac{\partial z}{\partial y}\cdot\dfrac{dy}{ds}$
### Backpropagation
![](https://i.imgur.com/TV4ENEh.png)

$C^n:$代表實際與預測的差距函數    
Loss Function即是總合了所有資料的差距$C^n$，但先從計算一筆資料開始，不看總合，
先考慮其中一個neuron(紅框處)    

### Backpropagation
![](https://i.imgur.com/XLmsxPF.png)

$z=x_1w_1+x_2w_2+b$    

$\dfrac{\partial C}{\partial w}=\dfrac{\partial z}{\partial w}\dfrac{\partial C}{\partial z}$
* $\dfrac{\partial z}{\partial w}:$forward pass
* $\dfrac{\partial C}{\partial z}:$backward pass

註：假設只有兩個input    

### Backpropagation - Forward pass
![](https://i.imgur.com/2xaurEQ.png)    
![](https://i.imgur.com/87cB7vg.png)    

* $\partial z/\partial w_1=x_1$
* $\partial z/\partial w_2=x_2$

forward pass的規律即是，input是什麼，微分之後就是什麼，即上層的output為下層的$\partial z/\partial w$    

如上圖範例所示，輸入是1與-1，其值為各自$\partial z/\partial w$的偏微分，以此類推。
### Backpropagation - Backward pass
![](https://i.imgur.com/XjYoOG7.png)    
![](https://i.imgur.com/W9A6HFz.png)    

$\dfrac{\partial C}{\partial z}=\dfrac{\partial a}{\partial z}\dfrac{\partial C}{\partial a}$
* $\dfrac{\partial a}{\partial z}:$sigmoid的微分(假設使用sigmoid)
* $\dfrac{\partial C}{\partial a}=\dfrac{\partial z'}{\partial a}\dfrac{\partial C}{\partial z'}+\dfrac{\partial z''}{\partial a}\dfrac{\partial C}{\partial z''}$

z通過activation function之後得到output，即$a=\sigma(z)$，a再做為其它neuron的輸入計算得到下一層的z....最後才得到C。    

### Backpropagation - Backward pass
![](https://i.imgur.com/JeWPzvR.png)    
![](https://i.imgur.com/z6SVsnC.png)    

$\dfrac{\partial C}{\partial z}=\sigma'(z)\left[ w_3\dfrac{\partial C}{\partial z'}+w_4\dfrac{\partial C}{\partial z''} \right]$    

換個角度來想反向傳播的計算過程，把上面的式子變成另一個neuron來看(如上二圖)。    

$\sigma'(z)$中的$z$是在計算前向傳播的時候就已經決定的值，因此它是一個常數項，並非像前向傳播的時候是一個非線性啟動函數。
### Backpropagation - Backward pass
![](https://i.imgur.com/0GioITt.png)    
![](https://i.imgur.com/KdZF6AJ.png)    

剩兩個項目待解：    
* $\dfrac{\partial C}{\partial z'}$
    * 如果是output layer
        * $\dfrac{\partial y_1}{\partial z'}\dfrac{\partial C}{\partial y_1}$
* $\dfrac{\partial C}{\partial z''}$
    * 如果是output layer
        * $\dfrac{\partial y_2}{\partial z''}\dfrac{\partial C}{\partial y_2}$

但是當這兩個項目不是屬於最後的輸出層時，那就必需要一直往下一層去推偏微分，直到最後的輸出層，求到輸出層的偏微分之後就可以將所有的解計算出來。
### Backpropagation - Backward pass
![](https://i.imgur.com/PaJsEp6.png)

實務上，我們並不會從輸入層開始計算反向傳播，而是從輸出層開始計算，一路往前推。
### Backpropagation - Summary
![](https://i.imgur.com/w7B007O.png)

總結來說，反向傳播就是將forward pass的偏微分跟backward pass的偏微分相乘即為解。
