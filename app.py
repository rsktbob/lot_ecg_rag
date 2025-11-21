from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os
from pathlib import Path

# 載入文件
def load_docs(folder="docs"):
    docs = []
    folder_path = Path(folder)
    
    if not folder_path.exists():
        raise FileNotFoundError(f"資料夾 '{folder}' 不存在！")
    
    # 載入 CSV 檔案
    csv_files = list(folder_path.glob("*.csv"))
    for p in csv_files:
        try:
            loader = CSVLoader(
                str(p), 
                encoding="utf-8",
                csv_args={ 'delimiter': ',' }
            )
            loaded_docs = loader.load()
            docs.extend(loaded_docs)
            print(f"  載入 CSV: {p.name} ({len(loaded_docs)} 筆資料)")
        except Exception as e:
            print(f"  無法載入 {p.name}: {e}")

    # 載入 PDF 檔案
    pdf_files = list(folder_path.glob("*.pdf"))
    for p in pdf_files:
        try:
            loader = PyPDFLoader(str(p))
            loaded_docs = loader.load()
            docs.extend(loaded_docs)
            print(f"  載入 PDF: {p.name} ({len(loaded_docs)} 頁)")
        except Exception as e:
            print(f"  無法載入 {p.name}: {e}")
    
    if not docs:
        raise FileNotFoundError(f"資料夾 '{folder}' 中沒有可用的 CSV 或 PDF 檔案！")
    
    return docs

# 切分文件
def split_docs(docs):
    if not docs:
        raise ValueError("沒有文件可以切分！")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
    )
    return splitter.split_documents(docs)


# 建立或載入向量庫
def build_retriever(splits, save_path="vectordb"):
    embeddings = OllamaEmbeddings(model="bge-m3")
    
    if Path(save_path).exists():
        print(f"  載入現有向量庫: {save_path}")
        vectordb = FAISS.load_local(
            save_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    else:
        print("  建立新向量庫...")
        vectordb = FAISS.from_documents(splits, embeddings)
        vectordb.save_local(save_path)
        print(f"  向量庫已儲存到: {save_path}")
    
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    return retriever


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 建立 RAG chain
def build_rag(retriever):
    # 問題重寫鍊
    contextualize_q_prompt = ChatPromptTemplate.from_messages([("system", """Given a chat history and the latest user question which might reference context in the chat history, 
                                                                formulate a standalone question which can be understood without the chat history. Do NOT answer the question, 
                                                                just reformulate it if needed and otherwise return it as is."""), 
                                                               ("human", "{chat_history}\n\nFollow Up Input: {question}")])
    contextualize_q_chain = contextualize_q_prompt | ChatOllama(model="llama3.1:8b", temperature=0.1) | StrOutputParser()


    # 問答鍊
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一個專業的助手。請根據以下參考資料回答問題。如果參考資料中沒有相關資訊，請誠實地說你不知道，不要編造答案，請使用繁體中文回答。
         參考資料：
        {context}"""),
        ("human", "{question}")
    ])
    qa_chain = {"context" : retriever | format_docs, "question" : RunnablePassthrough( )} | qa_prompt | ChatOllama(model="llama3.1:8b", temperature=0.1) | StrOutputParser()
    
    return qa_chain, contextualize_q_chain


if __name__ == "__main__":
    try:
        print("RAG 系統啟動中...")
        print("\n[1/4] 載入文件…")
        docs = load_docs("docs")
        print(f"  共載入 {len(docs)} 個文件")

        print("\n[2/4] 切分文件…")
        splits = split_docs(docs)
        print(f"  切分成 {len(splits)} 個片段")

        print("\n[3/4] 建立向量庫…")
        retriever = build_retriever(splits)

        print("\n[4/4] 建立 RAG chain…")
        qa_chain, contextualize_q_chain  = build_rag(retriever)
        chat_history = []

        print("\nRAG 系統準備就緒！")
        print("=" * 50)
        print("輸入問題開始對話（輸入 'exit'、'quit' 或 'q' 離開）")
        print("輸入 'rebuild' 重建向量庫")
        print("=" * 50)

        while True:
            query = input("\n你的問題: ").strip()
            
            if query.lower() in ["exit", "quit", "q", "離開"]:
                print("再見！")
                break
            
            if query.lower() == "rebuild":
                print("\n重建向量庫...")
                import shutil
                if Path("vectordb").exists():
                    shutil.rmtree("vectordb")
                docs = load_docs("docs")
                splits = split_docs(docs)
                retriever = build_retriever(splits)
                qa_chain, retriever = build_rag(retriever)
                print("向量庫重建完成！")
                continue
            
            if not query:
                continue

            try:
                print("\n思考中...")
                rewirtten_question = contextualize_q_chain.invoke({
                    "question":query,
                    "chat_history": "\n".join(chat_history)
                })

                # 使用 LCEL chain
                answer = qa_chain.invoke(rewirtten_question)
                
                print(rewirtten_question)
                # 獲取來源文件
                source_docs = retriever.invoke(rewirtten_question)

                chat_history.append(f"User: {query}")
                chat_history.append(f"Assistant: {answer}")

                print("\n回答:")
                print("-" * 50)
                print(answer)

                print("\n參考來源:")
                print("-" * 50)
                for i, doc in enumerate(source_docs, 1):
                    source = doc.metadata.get('source', '未知來源')
                    print(f"\n[{i}] 檔案: {Path(source).name}")
                    print(f"    內容: {doc.page_content[:150]}...")
                    
            except Exception as e:
                print(f"查詢時發生錯誤: {e}")

    except Exception as e:
        print(f"\n啟動失敗: {e}")
        print("\n請確認:")
        print("  1. Ollama 是否正在運行 (ollama serve)")
        print("  2. 模型是否已下載 (ollama pull llama3.1:8b 和 ollama pull bge-m3)")
        print("  3. docs 資料夾是否存在且有.csv檔案")