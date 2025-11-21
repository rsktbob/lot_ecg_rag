# RAG 系統（FAISS + Ollama + LangChain）

本專案實作一個簡易的 **RAG（Retrieval-Augmented
Generation）系統**，支援：

-   載入 **CSV 與 PDF**
-   使用 **RecursiveCharacterTextSplitter** 做文件切分
-   **FAISS** 做向量庫
-   使用 **Ollama Embeddings（bge-m3）** 作向量化
-   使用 **ChatOllama（llama3.1:8b）** 做回答
-   具備 **Contextualize Question（問題重寫）**\
    可將非獨立問題改寫成可檢索的完整問題

------------------------------------------------------------------------

## 1. 系統需求

### ● Python 3.10 或 3.11

### ● 安裝 Ollama

下載：https://ollama.com/download

啟動：

``` bash
ollama serve
```

------------------------------------------------------------------------

## 2. 必須下載的模型

### LLM 模型

``` bash
ollama pull llama3.1:8b
```

### 向量模型

``` bash
ollama pull bge-m3
```

------------------------------------------------------------------------

## 3. 創建 Python 環境

``` bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 4. 專案結構

    project/
    ├── docs/                # 放資料來源
    ├── vectordb/            # FAISS（執行後生成）
    ├── app.py               # 主程式
    ├── venv/                # python環境
    └── README.md
    

------------------------------------------------------------------------

## 5. 執行程式

``` bash
venv\Scripts\activate
python app.py
```

------------------------------------------------------------------------

## 6. 操作說明

  指令              作用
  ----------------- ------------------
  exit / quit / q   離開程式
  rebuild           刪除並重建向量庫

------------------------------------------------------------------------

## 7. RAG Pipeline

    User Query
       ↓
    Contextualize Question（問題重寫）
       ↓
    Retriever（FAISS）
       ↓
    文件
       ↓
    LLM 根據文件回答
