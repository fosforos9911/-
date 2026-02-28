from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import hashlib
import time  

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llms import get_embeddings

# 文件路径配置
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_JSON_PATH = BASE_DIR / "processing_results.json"
CHROMA_PERSIST_DIR = DATA_DIR / "chroma"
CHROMA_COLLECTION_NAME = "documents_collection"

# 全局变量，缓存向量储存
_vector_store: Optional[Chroma] = None


def load_processed_json() -> List[Dict[str, Any]]:
    """
    加载第一个py文件生成的JSON数据
    
    Returns:
        List[Dict]: 处理后的文档数据列表
        
    Raises:
        FileNotFoundError: 如果JSON文件不存在
    """
    if not PROCESSED_JSON_PATH.exists():
        raise FileNotFoundError(f"处理后的数据文件未找到: {PROCESSED_JSON_PATH}，请先运行 data_preprocess.py")
    
    with open(PROCESSED_JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"已加载 {len(data)} 个文档的JSON数据")
    time.sleep(2)  
    return data


def json_to_documents(json_data: List[Dict[str, Any]]) -> List[Document]:
    """
    将JSON数据转换为LangChain Document对象
    
    Args:
        json_data: 从第一个py文件获取的JSON数据
        
    Returns:
        List[Document]: 文档对象列表
    """
    documents = []
    
    for item in json_data:
        file_name = item['file_name']
        data = item['data']
        raw_text = data['raw_text']
        metadata = data.get('metadata', {})
        
        # 生成文档ID
        text_hash = hashlib.md5(raw_text.encode()).hexdigest()[:8]
        doc_id = f"{file_name}_{text_hash}"
        
        # 创建文档对象
        doc = Document(
            page_content=raw_text,
            metadata={
                'source': file_name,
                'doc_id': doc_id,
                'file_type': metadata.get('type', 'unknown'),
                'file_size': metadata.get('file_size', 0),
                **metadata  # 合并所有元数据
            }
        )
        documents.append(doc)
        
        # 如果有表格，将表格也作为单独的文档
        if data.get('tables'):
            for table_idx, table in enumerate(data['tables']):
                table_text = json.dumps(table, ensure_ascii=False, indent=2)
                table_doc = Document(
                    page_content=f"表格数据:\n{table_text}",
                    metadata={
                        'source': file_name,
                        'doc_id': f"{doc_id}_table_{table_idx}",
                        'file_type': metadata.get('type', 'unknown'),
                        'content_type': 'table',
                        'table_index': table_idx,
                        **metadata
                    }
                )
                documents.append(table_doc)
    
    print(f"转换了 {len(documents)} 个文档对象")
    time.sleep(2)  
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """
    将长文档拆分成更小的块
    
    Args:
        documents: 原始文档列表
        
    Returns:
        List[Document]: 拆分后的文档列表
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # 每个块最大字符数
        chunk_overlap=50,    # 块之间重叠字符数
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        length_function=len,
    )
    
    splits = text_splitter.split_documents(documents)
    print(f"文档已拆分为 {len(splits)} 个片段")
    time.sleep(2)  
    return splits


def get_vector_store(force_recreate: bool = False) -> Chroma:
    """
    获取向量存储实例，使用 Chroma 持久化
    
    Args:
        force_recreate: 是否强制重新创建向量库
        
    Returns:
        Chroma: 向量存储实例
    """
    global _vector_store
    
    # 如果已经初始化且不强制重建，直接返回
    if _vector_store is not None and not force_recreate:
        return _vector_store
    
    print("初始化向量存储...")
    time.sleep(1)  
    
    # 确保持久化目录存在
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    
    # 获取嵌入模型
    embedding_model = get_embeddings()
    
    # 检查是否需要重新创建
    should_recreate = force_recreate
    
    # 如果目录存在且有文件，尝试加载现有向量库
    if not should_recreate and CHROMA_PERSIST_DIR.exists() and any(CHROMA_PERSIST_DIR.iterdir()):
        try:
            _vector_store = Chroma(
                persist_directory=str(CHROMA_PERSIST_DIR),
                embedding_function=embedding_model,
                collection_name=CHROMA_COLLECTION_NAME,
            )
            # 检查集合是否为空
            if _vector_store._collection.count() > 0:
                print(f"从持久化目录加载向量存储: {CHROMA_PERSIST_DIR}，包含 {_vector_store._collection.count()} 个文档")
                time.sleep(2)  
                return _vector_store
            else:
                print("加载的向量存储为空，将重新创建")
                should_recreate = True
        except Exception as e:
            print(f"加载已存在的向量存储失败: {e}，将重新创建")
            should_recreate = True
    
    # 创建新的向量存储
    print("正在创建新的向量存储...")
    time.sleep(1)  
    
    # 加载JSON数据
    json_data = load_processed_json()
    
    # 转换为文档
    documents = json_to_documents(json_data)
    
    # 拆分文档
    splits = split_documents(documents)
    
    # 创建向量存储
    _vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory=str(CHROMA_PERSIST_DIR),
        collection_name=CHROMA_COLLECTION_NAME,
    )
    
    # 持久化
    return _vector_store


def search_documents(query: str, top_k: int = 5, filter_dict: Optional[Dict] = None) -> List[Document]:
    """
    检索相关文档
    
    Args:
        query: 查询文本
        top_k: 返回的文档数量
        filter_dict: 过滤条件，如 {"file_type": "pdf"}
        
    Returns:
        List[Document]: 相关文档列表
    """
    vector_store = get_vector_store()
    
    # 执行相似度搜索
    if filter_dict:
        docs = vector_store.similarity_search(query, k=top_k, filter=filter_dict)
    else:
        docs = vector_store.similarity_search(query, k=top_k)
    
    print(f"检索到 {len(docs)} 条相关文档")
    return docs


def search_with_score(query: str, top_k: int = 5, filter_dict: Optional[Dict] = None) -> List[tuple]:
    """
    检索相关文档并返回相似度分数
    
    Args:
        query: 查询文本
        top_k: 返回的文档数量
        filter_dict: 过滤条件
        
    Returns:
        List[tuple]: (文档, 相似度分数) 列表
    """
    vector_store = get_vector_store()
    
    # 执行带分数的相似度搜索
    if filter_dict:
        docs_with_scores = vector_store.similarity_search_with_score(query, k=top_k, filter=filter_dict)
    else:
        docs_with_scores = vector_store.similarity_search_with_score(query, k=top_k)
    
    print(f"检索到 {len(docs_with_scores)} 条相关文档")
    return docs_with_scores


def format_docs(docs: List[Document], include_metadata: bool = True) -> str:
    """
    格式化文档为字符串
    
    Args:
        docs: 文档列表
        include_metadata: 是否包含元数据
        
    Returns:
        str: 格式化后的文本
    """
    formatted = []
    
    for i, doc in enumerate(docs):
        content = doc.page_content.strip()
        
        if include_metadata and doc.metadata:
            source = doc.metadata.get('source', '未知来源')
            file_type = doc.metadata.get('file_type', '未知类型')
            header = f"[{i+1}] 来源: {source} ({file_type})"
            formatted.append(f"{header}\n{content}")
        else:
            formatted.append(f"{i+1}.{content}")
    
    return "\n\n" + "\n\n".join(formatted) + "\n"


def get_collection_stats() -> Dict[str, Any]:
    """
    获取向量数据库统计信息
    
    Returns:
        Dict: 统计信息
    """
    try:
        vector_store = get_vector_store()
        count = vector_store._collection.count()
        
        # 获取所有元数据中的文件类型分布
        file_types = {}
        all_docs = vector_store._collection.get(include=['metadatas'])
        if all_docs and all_docs['metadatas']:
            for meta in all_docs['metadatas']:
                if meta and 'file_type' in meta:
                    ft = meta['file_type']
                    file_types[ft] = file_types.get(ft, 0) + 1
        
        return {
            'total_documents': count,
            'collection_name': CHROMA_COLLECTION_NAME,
            'persist_directory': str(CHROMA_PERSIST_DIR),
            'file_types': file_types
        }
    except Exception as e:
        print(f"获取统计信息失败: {e}")
        return {'error': str(e)}


def rag_query(query: str, top_k: int = 3, filter_dict: Optional[Dict] = None) -> str:
    """
    基于RAG的文档查询入口函数
    
    Args:
        query: 用户查询文本
        top_k: 返回的文档数量
        filter_dict: 过滤条件
        
    Returns:
        str: 格式化后的文档内容
    """
    try:
        docs = search_documents(query, top_k=top_k, filter_dict=filter_dict)
        
        if not docs:
            return "未找到相关文档。"
        
        return format_docs(docs, include_metadata=True)
    
    except FileNotFoundError as e:
        print(f"数据文件未找到: {e}")
        return f"文档检索失败：{e}\n请先运行 data_preprocess.py 生成 processed_data.json 文件。"
    
    except Exception as e:
        print(f"RAG查询出错: {e}", exc_info=True)
        return f"文档检索时发生错误：{str(e)}"


def rebuild_vector_store():
    """
    强制重建向量数据库（用于数据更新后）
    
    Returns:
        bool: 是否重建成功
    """
    try:
        print("开始强制重建向量数据库...")
        time.sleep(1) 
        
        # 强制重新创建向量存储
        vector_store = get_vector_store(force_recreate=True)
        
        # 获取统计信息
        stats = get_collection_stats()
        print(f"向量数据库重建完成: {stats}")
        time.sleep(2)  

        return True
    except Exception as e:
        print(f"重建向量数据库失败: {e}")
        time.sleep(2)  
        return False


if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("向量数据库模块测试")
    print("=" * 60)
    time.sleep(2)  
    
    # 初始化向量库
    print("\n1. 初始化向量数据库...")
    time.sleep(1)  
    start = time.time()
    try:
        vector_store = get_vector_store()
        stats = get_collection_stats()
        print(f"初始化成功，耗时: {time.time()-start:.2f}秒")
        print(f"统计信息: {stats}")
        time.sleep(3)  
    except Exception as e:
        print(f"初始化失败: {e}")
        time.sleep(3)  
    
    # 执行几个查询
    print("\n2. 测试语义检索:")
    time.sleep(1)  
    
    test_queries = [
        "周一科技公司的注册资本是多少？",
        "周三酒店的客房出租率是多少？",
        "周二银行的净利润是多少？",
        "研发投入占比多少",
        "员工人数"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n查询{i}: {query}")
        time.sleep(1)  
        start = time.time()
        
        try:
            # 使用带分数的检索
            docs_with_scores = search_with_score(query, top_k=2)
            
            if docs_with_scores:
                for j, (doc, score) in enumerate(docs_with_scores, 1):
                    source = doc.metadata.get('source', '未知')
                    preview = doc.page_content[:80].replace('\n', ' ')
                    print(f"结果{j}: [{source}] 相似度: {1-score:.4f}")
                    print(f"预览: {preview}...")
                    time.sleep(1)  
            else:
                print(f"未找到相关文档")
            
            print(f"耗时: {time.time()-start:.2f}秒")
            time.sleep(1)  
            
        except Exception as e:
            print(f"查询失败: {e}")
            time.sleep(2)  
    
    # 过滤查询
    print("\n3. 测试过滤查询 (只检索PDF文件):")
    time.sleep(1)  
    try:
        docs = search_documents("酒店数据", top_k=2, filter_dict={"file_type": "pdf"})
        print(f"检索到 {len(docs)} 个PDF文档")
        for doc in docs:
            print(f"   - {doc.metadata.get('source')}")
            time.sleep(1)  
    except Exception as e:
        print(f"过滤查询失败: {e}")
        time.sleep(2)  
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    time.sleep(10) 