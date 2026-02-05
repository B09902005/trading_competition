# 簡介

這個專案包含我們在比賽期間，所使用的資料、模型訓練、以及策略的程式碼。

---

# 1. 環境設置

## 1.1 Python 環境
這個專案需要的版本為 **Python 3.10.15**。
```bash
# 建立 conda 環境
conda create -n myenv python=3.10.15

# 啟用環境
conda activate myenv

# 確認 Python 版本 (應該會印出 Python 3.10.15)
python --version
```

## 1.2 安裝套件
```bash
pip install -r requirements.txt
```


# 2. 使用方式

## 2.1 取得資料

### 執行方式
```bash
bash script/1_load_data.sh
```

### 功能
從 finlab 抓取最新的資料。若執行成功，在 data 資料夾的 .csv 檔案會更新至最新交易日的資料。

## 2.2 訓練模型

### 執行方式
```bash
bash script/2_train_model.sh
```

### 功能
對 data 的資料預處理後，訓練模型來進行預測。若執行成功，在 results 資料夾內會出現一個符合「{data\_category}\_{model\_name}\_{train\_test\_split}\_{loss}\_{data}」格式的資料夾，內含其對訓練資料與測試資料的預測。（主要是用到 whole_output.csv）

### 參數
- **test_year**: 希望讓模型預測哪一年。模型會使用該年份之前的資料來訓練。
    - 預設 2026。
    - 可選擇 2026、2025、2024、2023、2022、2021。
- **train_period**: 訓練資料要包含過去多少年的資料。
    - 預設 3。
    - 可選擇 1、2、3、4、5。
    - 例如，若 test_year=2026，train_period=3，則模型會使用 2023-2025 作為訓練資料，以 2026 年作為測試資料。
- **data_category**: 用來訓練和預測的股票池。
    - 預設 Top100。
    - 種類詳見 basic_info.json。
- **model_name**: 使用的模型。
    - 預設 iTransformer。
    - 可選擇 'iTransformer'、'DLinear'、'LSTM'、'TimesNet'、'CATS'。
- **loss**: loss function。
    - 預設 ConcordanceCorrelation。
    - 種類詳見 model.exp.exp_pct_prev.py 中的 loss_map。
- **data**: 資料預處理的方式。
    - 預設 Dataset_Individual_Seq_Norm。
    - 可選擇
         - Dataset_Individual_Seq_Norm：對該股票過去 T 天做標準化。
         - Dataset_GroupZ：資料取 log 後，對當天所有股票做標準化。
         - Dataset_Individual_Seq_Norm_with_GroupZ：資料取 log 後，對該股票過去 T 天做標準化，也對當天所有股票做標準化，並相加。


### 輸出
若順利執行，results 資料夾內，會出現一個「{data\_category}\_{model\_name}\_{train\_test\_split}\_{loss}\_{data}」格式的資料夾。在 whole_output.csv 中，可以看到包含「股票id、日期、模型預測漲幅、實際漲幅」的資料。

若在較新的資料中，發現實際漲幅那欄的數值小於 -1 是正常的，表示目前還無法得知 ground truth，（要在未來幾天後才會知道），故暫時用小於 -1 的數值表示。此不影響回測結果。


## 2.3 策略回測

### 執行方式
```bash
bash script/3_backtest.sh
```

### 功能
在模型訓練完畢後，拿模型的預測來跑策略並回測。若執行成功，在 outputs 資料夾中，會出現回測結果。

### 參數
- **backtest_start_year**: 回測起始年份。
    - 預設 2021。
    - 可選擇 2026、2025、2024、2023、2022、2021。
- **backtest_end_year**: 回測結束年份。
    - 預設 2026。
    - 可選擇 2026、2025、2024、2023、2022、2021。
- **train_period**、**data_category**、**model_name**、**loss**、**data**: 同模型訓練的參數，取決於想要回測哪套模型參數。只要是有訓練過對應的模型的參數組合都可以用。

該檔案會從 results 中，對應的參數設定裡，抓取模型的預測並將之轉為交易策略。如果參數設定錯誤，會印出 "no file {setting}.csv"，可以從這裡確認參數是否設定錯誤。

### 輸出
回測結果會放在 output 資料夾中，內有以下檔案：
- **returns.csv**: 照著策略執行交易的話，每天能夠持有的資產量。假設第一天為 10億。
- **current_stocks.csv**: 照著策略執行交易的話，在回測最後一天（或是最新的交易日），手上會持有的股票。一行表示一個倉位。一個倉位約佔總資產的 1/n，其中 n 為股票池的股票數。（若股票池為 Top100，則 n=100） 
- **trades.csv**: 照著策略執行交易的話，過去的買賣史。
- **positions.csv**: 回測程式的中間產物，表示模型對股票的樂觀程度。正為樂觀，負為悲觀，零為中立。
- **adx.csv**: 回測程式的中間產物，用來紀錄股價 ADX。

# 3. 交易方式

若要跑實際交易，將模型訓練時的 test_year 設定為 2026。回測時的 backtest_end_year 也設定為 2026。

每個交易日晚上十一點後（finlab 更新當日資料後），依序執行 script 資料夾內的三個 bash 檔案。若皆成功執行，會看到：

- 1_load_data.sh：data 資料夾內的 csv 檔案，出現當日的資料。
- 2_train_model.sh：results 資料夾，出現對應的設定參數的資料夾。資料夾內，test/whole_output.csv 裡面，會包含當天日期的預測。
- 3_backtest.sh：output 資料夾，current_stock.csv，裡面，last_date 會顯示當天日期。

觀察 output/current_stocks.csv，若出現 open_date 為隔天日期的股票，表示該策略預計於下一個交易日推薦買下的股票，原則上會在隔天開盤時，花約總資產的 1%，已開盤價買入。以下是 csv 檔案裡面，比較重點的內容：

- symbol：股票的編號
- open_date：買進日期
- last_date：隔天日期（所以如果 open_date 跟 last_date 一樣，表示隔天推薦買下這隻股票）
- open_price：買進價。原則上是會在開盤時用開盤價買進。
- last_price：目前的價格
- position_size：共幾股
- current_value：現在這個倉位的價值。即 position_size * last_price。

觀察 output/trades.csv，若出現 close_date 為隔天日期的股票，表示該策略預計於下一個交易日推薦賣出的股票。原則上會在隔天開盤時，將手中的該股票全數賣出。以下是 csv 檔案裡面，比較重點的內容：

- symbol：股票的編號
- open_date：買進日期
- close_date：賣出日期（所以如果 close_date 是隔天，表示隔天要賣出這支股票）
- open_price：買進價
- close_price：賣出價
- position_size：共幾股

# 4. 檔案概述

- script：包含要執行的 script 檔案。
- data：各個股票的資料。
- finlab：finlab 的暫時資料夾。
- model：內含模型訓練相關的程式碼。
    - exp：訓練架構的程式碼。
    - models：模型架構的程式碼。
    - trading_data_provider：資料預處理的程式碼。
    - utils：小工具。loss.py 有包含各種 loss 的計算方式。
    - result_log：純記錄用。
    - checkpoints：訓練完畢的模型存放處。
- results：模型的預測放在這裡。
- backtest：回測相關的程式碼。
- outputs：回測結果。
