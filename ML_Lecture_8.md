# 李宏毅_ML_Lecture_8
###### `Hung-yi Lee` `NTU` `Machine Learning`
[課程撥放清單](https://www.youtube.com/channel/UC2ggjtuuWvxrHHHiaDH1dlQ/playlists)
## ML Lecture 8-1: “Hello world” of deep learning
[課程連結](https://www.youtube.com/watch?v=Lx3l4lOrquw&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=13)      
### Mini-batch
![](https://i.imgur.com/0DQ7fH9.png)    
![](https://i.imgur.com/Aq9KKK8.png)    

實務上在執行深度學習的時候並不會真的去最小化『總(total)』的損失函數，而是將訓練資料集分成小批量(mini-batch)，每次以一小批量訓練優化，直到將所有的mini-batch都計算過，再迭代(epoch)相同的計算方式。    
    
因此，每看一個mini-batch就會更新一次參數，再乘上總epoch(迭代次數)即為參數的更新次數。    

如果mini-batch設置為1，那就是隨機梯度更新(Stochastic gradient descent)，其優點是速度較快，那為何要選擇mini-batch?    

### Speed
![](https://i.imgur.com/G5Bz3AO.png)    

以訓練資料集50000筆為例：    
* 隨機梯度(即batch-size=1)
    * 每次迭代更新50000次
    * 每次迭代166秒
* batch-size=10
    * 每次迭代更新5000次。    
    * 每次迭代17秒

可以發現，在隨機梯度完成一次迭代的時候，batch-size為10的部份已經可以完成幾乎十次的迭代了，兩者的參數更新次數也幾乎一樣，這時候當然會選擇batch-size較大，因為較為穩定。    

讓兩者之間幾無差異的原因在於gpu的平行運算，平行運算有其極限，故無法設置很大batch-size。    

另外一個batch-size無法設置很大的原因在於，實務上設置較大的batch-size無法有效的收斂，很大的機會會直接陷入區域最佳解，較小的batch-size可以利用隨機性的特值來跳脫那區域最佳解。    
## ML Lecture 8-2: Keras 2.0
[課程連結](https://www.youtube.com/watch?v=5BJDJd-dzzg&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=14)      
## ML Lecture 8-3: Keras Demo
[課程連結](https://www.youtube.com/watch?v=L8unuZNpWw8&index=15&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49)      

較多為實作說明，如有需求再點擊觀賞即可。