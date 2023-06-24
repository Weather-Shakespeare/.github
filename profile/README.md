# 天氣 Hackthon 沙士比亞

## 蛤！你怪獸怎麼一直變大（暫定）

我們是一群來自不同系所並且喜歡嘗試新事物的一群熱血份子。

將永續概念結合機器學習或圖像辨識，以提供碳追蹤的系統，最後實作在大家日常生活中常使用的通訊軟體 - LINE。我們秉持著與資訊工程結合的開源精神並且結合共筆概念管理團隊組織運作，像是個小型新創的超級新星。

在日常生活中，沒有落實回收可能對環境造成衝擊與汙染。相反地，回收並再製的產品相對由原料直接製成，可以減少大量能源消耗與製造的碳排。但身邊許多人時常為了方便沒有確實落實資源回收，從而導致產生額外的碳排與汙染。為了改善這種情況，我們設計了一個 Line bot，使用者可以拍下自身落實回收的照片，經過系統辨識後集點並養怪獸，鼓勵使用者確實做好垃圾分類，並讓使用者了解落實回收可以帶來的節約能源效益。

### 成員資訊

| 姓名                                         | 學校系所                   | 暱稱                   | 專長                                |
| -------------------------------------------- | -------------------------- | ---------------------- | ----------------------------------- |
| 葉霈恩（組長）                               | 中央大學大氣系三年級       | Deadline Superman      | Python, 簡報製作                    |
| 林群賀 [[GitHub]](https://github.com/1chooo) | 中央大學大氣系三年級       | 拖延大將軍             | Python, Web Dev, LINE BOT           |
| 黃品誠                                       | 中央大學物理學系三年級     | 大盲人                 | Python, Discord BOT, 碳足跡熱點分析 |
| 林源煜                                       | 中央大學大氣系四年級       | AI 小菜雞              | Python API 開發，機器學習           |
| 周姿吟                                       | 中央大學數學系計資組三年級 | Phd of procrastination | Matlab, LaTex, 影像處理             |

### 產品預計使用流程（模擬流程）
1. 使用者拍下回收物的照片並且回傳至 LINE BOT 聊天室
2. 後臺透過影像識別模型（我們會預先訓練）辨識回收物
3. 針對不同的回收物有不同的集點與怪獸加分機制（詳見實作方法 - 遊戲規則實作）
4. 整合回收減低碳排與碳足跡相關資訊

### 實作方法

#### 開發流程
1. 到便利商店、量販店、回收廠進行拍攝和通過爬蟲收集準備訓練的照片
2. 將照片歸檔分類並進行標籤處理（詳見**標籤類別**）
3. 將照片檔以 7:2:1 的比例歸類為訓練集、驗證集和測試集
4. 設計神經網絡模型並進行訓練
5. 訓練完成後將模型輸出
6. 開發 Line 聊天機器人，並將神經網絡模型合併入 Line Bot 當中

#### 影像辨識部分

1. 獲取日常生活中常見廢棄物的照片，並進行標籤化分類
2. 設計辨識廢棄物照片的分類器
3. 將圖片輸入分類器進行訓練
4. 持續優化模型訓練結果
5. 將完成訓練的模型輸出並部署
6. 持續性監控模型的效能，並在必要時重新訓練模型

<div align="center"> 
    <img src="https://hackmd.io/_uploads/B1wSqGN_3.png" width="500px;" alt=""/>
</div>


#### LINE BOT 實作部分
1. 利用 Line 的 Python Flask, LINE SDK Package 開發我們的 LINE BOT 產品
2. 通過 Line 提供的開發者界面設計聊天機器人的 UI
3. 將 Line Bot 和圖像辨識模型串接（開發可串接模型用之 API）
4. 實作會員卡功能

<div align="center"> 
    <img src="https://hackmd.io/_uploads/SJ4I9zV_n.jpg" width="500px;" alt=""/>
</div>

#### 遊戲規則實作 (Ho)

`(補充遊戲規則的圖)`

**回收機制、得分計算**
|       回收物       | 得分  | 回收物數量累積           | 得分 | 連續回收天數           | 得分 |
| :----------------: | :---: | -------------------- | ---- | ---------------------- | ---- |
| 回收一個紙杯、紙碗 |   1   | 回收物數量累積十個   | 5    | 連續回收天數累積七天   | 5    |
|   回收一個寶特瓶   |   2   | 回收物數量累積一百個 | 10   | 連續回收天數累積三十天 | 10   |
|   回收一個鋁箔包   |   3   | 回收物數量累積五百個 | 20   | 連續回收天數累積一百天 | 20   |



**成就系統算分**
|                    | 銅牌  | 銅牌  | 銀牌  | 金牌  |
| :----------------: | :---: | :---: | :---: | :---: |
|  回收物數量（個）  |   1   |  10   |  100  |  500  |
| 連續回收天數（天） |   1   |   7   |  30   |  100  |


#### 整合平台實作
`放我們會用到的資訊、方法，全台垃圾、回收處理資訊`

- 資料來源
  1. 圖解生活中的碳足跡－蘇峻葦(書，資料來源待查)
  2. 環保署環境開放資料平台
  3. 綠色光譜媒合資訊平台
- 可整合資訊
  1. 常見一次性產品的碳足跡
  2. 各縣市一般廢棄物回收率
  3. 公告列管材質回收率

### The Tech we will use

[[VIEW `developing_note.md` ]](./developing_note.md)

### Potential Problem 

<details>
<summary><strong>Q: 圖片識別情況：</strong></summary>

- A: 因為回收物普遍會壓扁，所以可能收到的照片不具備原有的特徵
</details>
<details>
<summary><strong>Q: 使用者連續傳送同一張圖片 :</strong></summary>

- A: 利用矩陣運算判斷兩張圖的 pixel value 差異值，若差異小於設定的 tolerance, 回傳警告語給用戶
</details>
<details>
<summary><strong>Q: 使用者傳送同回收物但不同背景的圖片 :</strong></summary>

- A: 先幫用戶加分，事後以人工方式過濾，排除短時間傳送同回收物的情況；或是利用影像分割技術判斷兩張圖片的回收物相似度多少，若高度相似則回傳用戶「疑似為同個回收物」
</details>
<details>
<summary><strong>Q: 判斷不準確 :</strong></summary>

- A: 前期在資料量不太充足時，可以有個用戶確認機制。
</details>
<details>
<summary><strong>Q: 後續點數兌換 :</strong></summary>

- A:
    1. 前期先以解鎖頭貼 or 角色人物為主
    2. 後期可以和商家談合作(以會產生大量回收物的商家為主，例如手搖飲店)
    3. 或是和民營的大眾運輸談合作（如果我們可以證明回收所帶來的減少碳足跡之效益，可以把 LINEBOT 上的集點點數轉換成 搭乘大眾運輸/騎共享單車 的優惠額度，變相鼓勵民眾搭乘大眾運輸，減少其他方面的汙染來源）
</details>
<details>
<summary><strong>Q: 如何增加用戶使用頻率 :</strong></summary>

- A: 
    1. 讓使用者體會到產品服務的價值，避免出現「平台/LineBot 使用介面過度複雜」的情況，讓使用者好上手，例如：拍宣傳/使用流程短片
    2. 鼓勵用戶評論回饋，收集第一手資料有利於後續介面或功能的優化，並解決用戶所提出的問題
    3. 發展後期可以提供用戶「個人化」的體驗 (但具體的體驗可以透過市場調查 或是 內部討論)
    4. 可以設置每日打卡活動，例如：連續登入多天解成就 or 連續多天閱讀平台上的文章資訊可以另外有獎勵機制
</details>

### Reference

<details><summary>VIEW MORE</summary>
<p>

- [Chop Shop (Open Source)](https://github.com/coryl/ChopShopOpenSource/blob/master/README.md)
- [OpenCV GrabCut: Foreground Segmentation and Extraction](https://pyimagesearch.com/2020/07/27/opencv-grabcut-foreground-segmentation-and-extraction/)
- [Interactive Foreground Extraction using GrabCut Algorithm](https://docs.opencv.org/3.4/d8/d83/tutorial_py_grabcut.html)
- [Green togo 綠色光譜廢棄物媒合平台](https://www.green-togo.tw/green-footprint.php)
- [低碳生活提案(圖解生活中的碳足跡)](https://vocus.cc/lowcarbon/home)
- [YOLO - object detection](https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html)
- [資源回收效益分析 NCTU，鍾佩樺，2013](https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html)
- [Image Segmentation with Diffusion Probabilistic Models](https://arxiv.org/pdf/2112.00390.pdf)
- [OpenCV Python 中文教學](https://github.com/grady1006/OpenCV-Python-Chinesse-Tutorials)
- [Machine Learning Model Deployment](https://www.dominodatalab.com/blog/machine-learning-model-deployment)
- [【python】opencv 2小時初學者教學 ｜ 影像辨識 ｜ 影像處理 ｜ 人臉辨識 ｜ 電腦視覺](https://www.youtube.com/watch?v=xjrykYpaBBM)
- [OpenCV-Python——第26章：SIFT特征点提取算法](https://blog.csdn.net/yukinoai/article/details/88912586)
- [圖像特徵比對(一)-取得影像的特徵點](https://chtseng.wordpress.com/2017/05/06/%E5%9C%96%E5%83%8F%E7%89%B9%E5%BE%B5%E6%AF%94%E5%B0%8D%E4%B8%80-%E5%8F%96%E5%BE%97%E5%BD%B1%E5%83%8F%E7%9A%84%E7%89%B9%E5%BE%B5%E9%BB%9E/)
- [A Generalized Image Classifier based on Feature Selection](http://rportal.lib.ntnu.edu.tw:8080/server/api/core/bitstreams/7160b71e-2ffe-4768-bd35-9dad23365c19/content)
- [Image Retrieval For Buildings and Scenic Spots](https://cv.cs.nthu.edu.tw/upload/undergraduate/9662141/index.htm)
- [[OpenCV]基礎教學筆記：影像讀取、前處理(with python)-001](https://medium.com/jimmy-wang/opencv-%E5%9F%BA%E7%A4%8E%E6%95%99%E5%AD%B8%E7%AD%86%E8%A8%98-with-python-d780f571a57a)
- [[Day 28]特徵擷取](https://ithelp.ithome.com.tw/articles/10304504?sc=iThelpR)
- [用Python實現OpenCV特徵提取與圖像檢索 | Demo](https://kknews.cc/zh-tw/code/r5394eo.html)

</p>
</details>
