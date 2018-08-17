# 李宏毅_ML_Lecture_6
###### `Hung-yi Lee` `NTU` `Machine Learning`
[課程撥放清單](https://www.youtube.com/channel/UC2ggjtuuWvxrHHHiaDH1dlQ/playlists)
## ML Lecture 6: Brief Introduction of Deep Learning
[課程連結](https://www.youtube.com/watch?v=Dr-WRlEFefw&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=11)  
### Deep learning attracts lots of attention.
![](https://i.imgur.com/RqjLG7e.png)    

上圖是google專案中應用到深度學習的趨勢圖，從2012年幾乎0到2016年逐步成長。    
### Ups and downs of Deep Learning
![](https://i.imgur.com/5OUChLZ.png)    

上圖是深度學習的近代史，從Perceptron的提出，限制、多層感知器、反向傳播....一直到現在。    
### Three Steps for Deep Learning
![](https://i.imgur.com/ijuhH7e.png)    

深度學習的步驟與機器學習雷同，第一步是定義Model，而Model就是上節課Logistic最後所提的，將很多個Logistic串起來而成，每一個Logistic有它自己的Weight，其中一個方式稱為Fully Connect Feedforward Network
### Fully Connect Feedforward Network
![](https://i.imgur.com/asXpKxB.png)    
![](https://i.imgur.com/P9Gp6N8.png)    
![](https://i.imgur.com/HHCsE5M.png)    
![](https://i.imgur.com/skdlmz4.png)    


將所有的節點通通連接在一起，並各自擁有權重，輸入1與-1之後最後經過多個sigmoid所得的輸出為0.62與0.83。    
在已知權重情況下，將每一個Neuron都視為一個function，在未知權重情況下，我們就是定義一個function set，透過學習讓它自己取得自己的權重。    
### Fully Connect Feedforward Network
![](https://i.imgur.com/4rNT1Ov.png)    

上圖是神經網路的架構，neuron之間互相連接，故稱為Fully Connect，計算由前往後，故稱為Feedforward Network。    

名稱定義：    
* 輸入：input layer
* 輸出：output layer
* 中間：hidden layers

### Deep
![](https://i.imgur.com/MsZDRq0.png)    

上圖是各種架構神經網路的組成，從AlexNet的8層到Residual Net的152層。    

### Matrix Operation
![](https://i.imgur.com/QywwcGZ.png)    

上圖說明以Matrix Operation來表示神經網路，也說明著在計算神經網路的時候如果透過矩陣計算做批次性的運算。    
### Neural Network
![](https://i.imgur.com/UgsxQPy.png)    
![](https://i.imgur.com/Mt4pIA1.png)    

上圖說明神經網路一層一層的向後計算，就是一連串的Matrix與Vector的計算    
$\sigma(W^La^{L-1}+b^L)$    
* 輸入層可以視為$a^0$
* $\sigma$：啟動函數
### Output Layer
![](https://i.imgur.com/pk3qF16.png)    

hidden layer可以視為特徵擷取，以此取代之前所做的特徵工程，讓神經網路自己學習，最後的output layer再利用softmax做類別輸出。    
### Example Application
![](https://i.imgur.com/3C3D8P6.png)    
![](https://i.imgur.com/WaMTmh1.png)    
![](https://i.imgur.com/NxMgJ73.png)    

舉例說明，輸入一張照片經過機器判斷它的數值，照片大小為16x16。    
* 16x16=256：代表輸入有256維的向量
* 有值為1，空白為0
* 輸出0-9：代表輸出有10維的向量
### FAQ
![](https://i.imgur.com/g74Qxwp.png)    

對於該如何決定layer與每一層layer的neuron數量，這只能靠直覺與經驗不斷的測試來取得。    

從傳統機器學習到深度學習，只是從一個問題跳到另一個問題，以機器視覺為例，傳統機器學習可能需要對照片做特徵工程，但深度學習可以直接輸入，不過相對的要決定layer與neuron。    

實務上的應用就必需考量，影像、語音辨視較難以抽取特徵，那就讓它自己學習，其餘的部份就case by case。    

### Loss for an Example
![](https://i.imgur.com/A4twm6D.png)    
![](https://i.imgur.com/CdzpxuL.png)    
![](https://i.imgur.com/zwWC290.png)    

Cost Function一樣是最小化Cross entropy為目標，而最佳化所使用的還是Gradient Descent    
$C(y, \hat{y})=-\sum\hat{y}_ilny_i$    
### Backpropagation
![](https://i.imgur.com/jLPkNUc.png)    

目前backpropagation都交由framework來處理。
### Universality Theorem
![](https://i.imgur.com/5SJAvY8.png)    

有一個理論，只要neuron夠多，不需要深也可以達到相同的目的。(後續說明)
### Resource
![](https://i.imgur.com/ScWNzDS.png)    

上面是老師提供的學習資源
