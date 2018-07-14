# 李宏毅_ML_Lecture_0
###### `Hung-yi Lee` `NTU` `Machine Learning`
[課程撥放清單](https://www.youtube.com/channel/UC2ggjtuuWvxrHHHiaDH1dlQ/playlists)
## ML Lecture 0-1:Introduction of Machine Learning
[課程連結](https://www.youtube.com/watch?v=CXgbekl66jc&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49)
### ARTIFICIAL INTELLIGENCE
![](https://i.imgur.com/6LAFPfe.png)

人工智慧是目標，而機器學習是實現人工智慧的一個手段，深度學習則是機器學習的一環。    
機器可以達到人工智慧跟生物一樣，先天的本能(人類設置好的)或是後天的學習。
### 人類設定好的天生本能
![](https://i.imgur.com/kr9a8OH.png)

以chat bot來看，如果只能按照程式設置的邏輯來走，那就沒有辦法執行邏輯之外的指令。    
設置『turn off』來關閉一個設備，即使你告訴它『don't turn off』，它依然會因為聽到關鍵字『turn off』而關閉設備。
### 人類設定好的天生本能
![](https://i.imgur.com/roeLwFz.png)
圖片是Yann LeCun在fb上分享的照片，號稱AI的產品切開來看，結果裡面藏了一堆『IF』的判斷式。
### What is Machine Learning
![](https://i.imgur.com/cdizDoS.png)

機器學習，即是找一個function來處理資料，語音辨視、視覺辨視...等，這個function即為model，給了訓練資料(input)，回饋需求資訊(output)，但你需要想辦法找出一個最好的function，那就需要一個演算法來執行，幫我們找到那個最好的function。
### Framework
![](https://i.imgur.com/4Yn9RT9.png)
以影像辨視為例，我們要先有function set(成千上萬的function即為Model)，假設func1可以順利的辨視貓跟狗，而func2將貓辨視為猴子，狗辨視為蛇，那func2當然是不好的。    
接著要有『Training Data』(訓練資料)，讓機器知道什麼照片是什麼相對應的輸出，這也是監督式學習。
最後要從這成千上萬的function set中透過Testing來找出一個最好的function。
### Learning Map
![](https://i.imgur.com/NJv6Zrp.png)

課程中會學到的東西：
* Supervised Learning
    * Regression
    ![](https://i.imgur.com/scwXH1g.png)
        * 預測為數值型
    * Classification
    ![](https://i.imgur.com/dEOJY9e.png)
        * 預測為類別型
            * 二元分類
                * 垃圾郵件分類
            * 多元分類
                * 新聞類型分類(政治、經濟、體育、娛樂...)
* Unsupervised Learning
![](https://i.imgur.com/NtDRZM6.png)
    * No label
    * 讓機器自己學習出東西
* Structured Learning
![](https://i.imgur.com/8Ict2MJ.png)
    * 輸出是一個複雜的結構性物件
* Reinforcement Learning
![](https://i.imgur.com/LixBziI.png)
    * 不同於supervised，不給答案，給它打分數。
    * 沒有足夠的資料做Supervised Learning的時候才做。

Supervised Learning:    
1. Input,Ouput之間是有關聯的
2. Ouput:即為label
Semi-supervised Learning:    
1. 僅部份資料有label
Unsupervised Leraning:    
1. 無label
## ML Lecture 0-2: Why we need to learn machine learning?
[課程連結](https://www.youtube.com/watch?v=On1N8u1z2Ng&index=2&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49)
這節課李宏毅老師利用寶可夢來說明AI訓練師，輕鬆逗趣。