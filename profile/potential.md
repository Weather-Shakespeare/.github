# Ppotential Problem


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