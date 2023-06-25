# Developing Note

### OpenCV

#### 影像讀取與前處理

##### 通過 OpenCV 讀取 jpg 檔案

```python
img = cv.imread('files.jpg')
cv.imshow('Cats',img)
cv.waitkey(0)
```
##### 利用 matplotlib 顯示影像

```python
blank = np.zeros((500,500,3),dtype='uint8')
blank[200:300, 300:400] = 255,255,100
cv.imshow('Cyan', blank)
#2. 畫方框(畫布, 起點, 終點, 顏色, 粗度)
# 如果 thickness=-1則為填滿
cv.rectangle(blank, (0,0), (250,250), (0,255,0), thickness = 3)
cv.imshow('Green', blank)
# 3. 畫圓形(畫布, 圓心位置, 半徑radius, 顏色, 粗度)
# 如果 thickness=-1則為填滿
cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 40, (0,0,255), thickness = 3)
cv.imshow('Red', blank)
# 4. 畫線條(畫布, 起點, 終點, 顏色, 粗度)
cv.line(blank ,(0,0), (blank.shape[1]//2, blank.shape[0]//2), (255,255,255), thickness = 3)
cv.imshow('Red', blank)
# 5. 加入文字(畫布, 文字內容, 起點, 字體, 大小, 顏色, 粗度)
cv.putText(blank, 'This is Jimmy.', (150,225), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0,255,0), 2)
cv.imshow('Text', blank)
# 等待圖片關閉
cv.waitKey(0)
```

##### 圖像預處理
包含模糊化、輪廓化、膨脹和侵蝕，不同的組合和處理順序有不同的效果。

```py
img = cv.imread('../Resources/Photos/cats.jpg')
# 糊糊化(blur)
# 注意 kernel size必須是奇數(odd)，kernel size越大越模糊。
# 模糊化也有許多不同的方法可以進行，讓 kernel依據需求進行運算。
blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)
cv.imshow('Blur Cats', blur)
# 輪廓化(edge cascade)
# 小技巧，可以先將圖片模糊化，再進行輪廓化，可以抓到比較少雜訊。
canny = cv.Canny(blur, 125, 175)
cv.imshow('Canny Cats', canny)
# 注意膨脹、侵蝕用的照片是已經輪廓化處理過。雜訊會較少。
# 膨脹 dilating
dilated = cv.dilate(canny, (7,7), iterations=3)
cv.imshow('Dilated', dilated)
# 侵蝕 eroding
eroded = cv.erode(canny, (3,3), iterations=1)
cv.imshow('Eroded', eroded)
cv.waitKey(0)
```

更進階的圖像處理方法：  
**Opening (侵蝕後膨脹) **和 **Closing（膨脹後侵蝕）**

##### 圖像旋轉
```py
img = cv.imread('../Resources/Photos/cats.jpg')
# 旋轉圖片
def rotate(img, angle, rotPoint=None):
 (height,width) = img.shape[:2]
if rotPoint is None:
 rotPoint = (width//2,height//2)
 
 rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
 dimensions = (width,height)
return cv.warpAffine(img, rotMat, dimensions)
# 輸入角度作為參數
rotated = rotate(img, -45)
cv.imshow('Rotated', rotated)
rotated_rotated = rotate(img, -90)
cv.imshow('Rotated Rotated', rotated_rotated)
```

##### 圖像結構
- 輪廓：沿著物件邊緣搜尋邊緣像素所形成的路徑
- 輪廓搜尋：根據二維影像搜尋影像中物件的輪廓，並以特定的資料結構儲存，紀錄輪廓的相關資料（階層式儲存）。
- 外部輪廓：影像的外緣
- 內部輪廓：若物件內包含洞，該物件在洞的邊緣所形成的輪廓

```py
img = cv.imread('../Resources/Photos/cats.jpg')
blank = np.zeros(img.shape, dtype='uint8')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
canny = cv.Canny(blur, 125, 175)

contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
# 計算總共有幾個輪廓 contours
print(f'{len(contours)} contour(s) found!')
# 畫出當前所有的 contours
cv.drawContours(img, contours, -1, (255,0,0), 1)
cv.imshow('Contours Drawn on img', img)
# 標示 contours
cv.drawContours(blank, contours, -1, (0,255,0), 1)
cv.imshow('Contours Drawn on blank', blank)
cv.waitKey(0)
```

##### 特徵拮取
根據二維的數位影像拮取一維的影像特征資料（特征向量），牽涉降維度運算（如PCA）
基本的影像識別步驟：包含 Keypoint detection, feature extraction, feature matching
* Keypoint detection: 在圖片中取得感興趣的關鍵點
* Feature extraction: 針對各關鍵點提取該區域的特征
* Feature matching: 關鍵點篩選並進行匹配

- 步驟：
    - 在圖片 A 與 B 分別找出 Keypoint 
    - 依據各個 keypoint 取出其 local features
    - 兩張圖片的 local features 進行 feature matching
    - 計算各個距離最小 keypoint

- Keypoint detection
有多種 Keypoint detector 的演算法有很多，OpenCV 中有 11 種。演算法之間可以組合使用，不同的情境適合不同的 detector。以下介紹最簡單的 FAST detector。
- Fast detector
    - 原理簡單
    - 運算速度快
    - 用於偵測 corners
    - 適用於即時偵測
    - 適用於低階執行環境
    - 最被廣泛使用的偵測器

```py
detector = cv2.FeatureDetector_create("FAST")
kps = detector.detect(gray)
print("# of keypoints: {}".format(len(kps)))
```

### 通用影像分類器
- 通過 SVM 進行分類
    - 直接找到一個決策邊界，使得類別與類別之間的距離能夠最大化
    - 通過 RBF 進行計算
- 影像特征
    - 色彩直方圖：圖像在色譜的分佈狀況
    - SIFT：偵測有可能的興趣點和拮取關鍵的描述子
    - Visual Bag of Words：將拮取的興趣點投影到128維的空間，通過K-means進行分群，形成特征集合
- 特征選取
    - Filter method: 利用特征本身的計算權重，經由特征集合挑選權重值高的特征子集合作為最後的訓練特征
    - Wrapper method: 通過最佳化演算法將能夠提升準確率的特征納入特征子集合
    - Filter + Wrapper: 先將相差甚遠的特征進行過濾，再利用Wrapper找出最佳的特征子集合

### [OpenCV-Python——第26章：SIFT特征点提取算法](https://blog.csdn.net/yukinoai/article/details/88912586)
- SIFT 特徵 (focus 在局部特徵) 算法可以處理兩圖像之間的平移、旋轉、invariant transform 的匹配問題
- 少量的物體即可產生大量 Sift eigen vector
- improved SIFT Algorithm 可以達到 real-time
- 經此算法所找到的 Key point (我猜是 eigen value or specific pixel), 不易受光照、invariant transform、noise 等因素影響. 特徵點例如: edge point
- 算法步驟
    - **Step 1. Scale space中的極值搜索** 利用 scale-space filtering
        $$D(x,y,\sigma)=(G(x,y,k\sigma)-G(x,y,\sigma))*I(x,y)=L(x,y,k\sigma)-L(x,y,\sigma)$$
    DoG(Difference of Gaussian) 利用兩個不同的 $\sigma$ 完成 filtering. 
    $*$: convolution 
    $G(x,y,\sigma)$: 變化尺度的 Gaussian function,
    $$G=\frac{1}{2\pi\sigma^2}e^{-\frac{(x-m/2)^2+(y-n/2)^2}{2\sigma^2}}$$
    **Remark:** $m,n$ is determined by $(6\sigma+1)(6\sigma+1)$ (可以想像成是維度)(具體的需要再查); $\sigma$ 是尺度空間因子, the smaller $\sigma$ value, the less smooth of image; 大$\sigma\rightarrow$ 圖像結構, 小 $\sigma\rightarrow$ 細節紋理
    $I(x,y)$: original image
    **Key point (最能代表這個 scale 的點): 得到 DoG 之後, 將圖像中某個 pixel, 與其附近的 8 neighbor pixels 比較, 且與同 Octave 的 previous scale & next scale 的 9 個 pixels 比較. If this pixel is a local extrema, then we say this is a key point.**
    - **Step 2. 確認 key point** 利用 *Hessian matrix* 得到 eigen value $\alpha, \beta$ 分別表示圖像 x,y 方向的梯度 (詳細公式看原文). 因 DoG 會導致較強的邊緣響應點(?? 猜測是增強 edge of object), 為了移除這類的點, 需設置閾值(在原文), 將滿足條件的 key point 保留. 低對比度和邊緣的  key point 在此步驟被丟掉.
    - **Step 3. 確認梯度方向** 利用圖像的 local gradient, 給每個 key point 一個或多個方向, 以利後續根據這些方向進行 tramsform. key point 的主方向是取其 neighborhood 梯度方向峰值最大的, 輔方向則保留峰值 $>$ 主方向峰值 80% 的方向.
    So far, 圖像的 SIFT 特徵點是涵蓋三個資訊的 key point: 
      1. 位置
      2. 尺度scale
      3. 方向
    - **Step 4. 關鍵點描述** 變換梯度的表示方法, 有利呈現較大的局部變形和光照變化. 梯度由來: 在每個 key point 的 neighborhood 並選定 scale 而測量得到 圖像的 local gradient.

**例子:**

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('')
img1 = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray, None) # gray: one channel

cv2.drawKeypoints(gray, kp, img)
cv2.drawKeypoints(gray, kp, img1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) #flags 有四種模式

plt.subplot(121), plt.imshow(img),
plt.title(''), plt.axis('off')
plt.subplot(122), plt.imshow(img1),
plt.title(''), plt.axis('off')
plt.show()
```

### [【python】opencv 2小時初學者教學 ｜ 影像辨識 ｜ 影像處理 ｜ 人臉辨識 ｜ 電腦視覺](https://www.youtube.com/watch?v=xjrykYpaBBM)
- 基礎指令

```python
cv.waitkKey(1000) # 等待1000毫秒才關閉, i.e. 1秒鐘; 設定讓圖片展示的時長
cv.waitkKey(0) # 直到按下鍵盤任意鍵 or 主動打叉圖片, 才會關閉圖片

# 改變圖片大小: (1)指定大小 (2)指定縮放倍數
cv2.resize(img, (300, 300)) 
cv2.resize(img, (0, 0), fx=0.5, fy=0.5) 
``` 

- 如果要檢測輪廓(edge detection), 轉成 gray scale image (no need rgb), 好處是可以減少運算量(3 channel $\to$ 1 channel) 

```python
# 檢測邊緣
canny = cv2.Canny(img, 150, 200) #設定門檻的最小最大值

# 偵測輪廓: 回傳輪廓 & 階層, 階層用不到
# 選擇偵測模式: 內輪廓 or 外輪廓 or 內外都要; 選擇近似方法: 可以壓縮水平 or 垂直方向的輪廓點
contours, hierarchy = cv2.fingContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
```

### [Machine Learning Model Deployment](https://www.dominodatalab.com/blog/machine-learning-model-deployment)
![](https://hackmd.io/_uploads/B10TeaWO3.png)
- **Training:**  select an algorithm, set its parameters and train it on prepared, cleaned data
- **Validation:** includes testing the model on a fresh data set and comparing the results to its initial training.
- **Deployment:** model needs to be moved into its deployed environment $\to$ the model needs to be integrated into a process $\to$ 使用模型的人員需要接受培訓，以了解如何啟動模型、訪問其數據和解釋其輸出
- **Monitoring:** The best way to monitor a model is to routinely evaluate its performance in its deployed environment. Every deployed model has the potential to degrade over time due to such issues as:
    1. Variance in deployed data. The data given to the model in deployment is not cleaned in the same manner as the training and testing data were.
    2. Changes in data integrity.
    3. Data drift. 人口統計變化、市場變動等等可能會隨時間的推移而改變
    4. Concept drift. 終端用戶對於何為正確預測的期望可能會發生變化，使得模型的預測不再具有相關性

### [Image Segmentation with Diffusion Probabilistic Models](https://arxiv.org/pdf/2112.00390.pdf)
- diffusion probabilistic method 被用於圖像生成。此論文提出了一種擴展這種模型以進行圖像分割的方法。該方法無需依賴 pre-trained backbone。將輸入圖像的信息和當前分割圖的估計結果相加，通過兩個編碼器的輸出進行合併。然後使用額外的編碼層和解碼器迭代地改進分割圖，使用擴散模型。由於擴散模型具有概率性質，因此它被應用多次，將結果合併為最終的分割圖。
-  including active contour and their deep variants, encoder-decoder architectures, and U-Nets
-  用到的 dataset: 
    1. the Cityscapes validation set 
    2. the Vaihingen building segmentation benchmark 
    3. the MoNuSeg dataset.
![](https://hackmd.io/_uploads/Hy_aPpWd3.png)

對於使用在擴散模型中的 UNet，我們進一步進行解耦。對於其編碼器，我們將其進一步拆分為 E、F、G 三部分。其中，E 負責與解碼器 D 進行連接，G 負責編碼原始圖像，F 負責編碼前一步的噪聲。

$$\epsilon_{\theta}(x_t, I, t) = D(E(F(x_t) + G(I), t), t). (17)$$

### [YOLO - object detection](https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html)

an extremely fast multi object detection algorithm (uses **CNN**)
- Step 1. Load the YOLO network
- Step 2. Create a blob (the input to the network is a so-called blob object): it has the following parameters:
    - the image to transform
    - the scale factor (1/255 to scale the pixel values to [0..1])
    - the size, here a 416x416 square image
    - the mean value (default=0)
    - the option swapBR=True (since OpenCV uses BGR)
- Step 3. Identifiy objects
- Step 4. 3 Scales for handling different sizes. The YOLO network has 3 outputs:
    - 507 (13 x 13 x 3) for large objects
    - 2028 (26 x 26 x 3) for medium objects
    - 8112 (52 x 52 x 3) for small objects
- Step 5. Detecting objects