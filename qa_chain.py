import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document



def process_question(question: str) -> str:
    """
    问题输入处理函数 
    
    Args:
        question: 原始问题文本
        
    Returns:
        str: 处理后的问题文本，直接用于向量检索
    """
    if not question or not isinstance(question, str):
        return ""
    
    # 去除首尾空格
    cleaned = question.strip()
    
    # 去除多余空格
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # 去除特殊字符
    cleaned = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', cleaned)
    
    # 如果问题没有问号，添加问号
    if not (cleaned.endswith('?') or cleaned.endswith('？')):
        cleaned = cleaned + '？'
    
    return cleaned


def process_question_simple(question: str) -> str:
    """
    极致简化的版本 
    
    Args:
        question: 原始问题文本
        
    Returns:
        str: 处理后的问题文本
    """
    return question.strip() if question else ""



# 提示词模板
DETAILED_TEMPLATE = """你是一个专业的文档问答助手。请严格基于提供的上下文信息回答问题。

【上下文信息】
{context}

【用户问题】
{question}

【回答要求】
1. 答案必须基于上下文信息，不要编造
2. 如果上下文中没有相关信息，请说"根据提供的文档，无法回答该问题"
3. 答案要简洁准确，避免多余信息
4. 如果涉及数据，可以直接引用

【答案】"""


def get_prompt_template() -> str:
    """
    获取提示模板字符串
    
    Returns:
        str: 详细的提示模板
    """
    return DETAILED_TEMPLATE


def create_prompt() -> ChatPromptTemplate:
    """
    创建LangChain提示模板
    
    Returns:
        ChatPromptTemplate: LangChain提示模板对象
    """
    return ChatPromptTemplate.from_template(DETAILED_TEMPLATE)


def create_qa_prompt(question: str, context: str) -> ChatPromptTemplate:
    """
    创建包含具体内容的问答提示
    
    Args:
        question: 用户问题
        context: 上下文信息
    
    Returns:
        ChatPromptTemplate: 填充后的提示模板
    """
    prompt = ChatPromptTemplate.from_template(DETAILED_TEMPLATE)
    return prompt



def retrieve_documents(
    query: str,
    vector_store,
    top_k: int = 5,
    filter_dict: Optional[Dict] = None
) -> List[Document]:
    """
    从向量数据库检索相关文档 - 最简单常用的方式
    
    Args:
        query: 查询文本（已处理过的问题）
        vector_store: Chroma向量存储实例
        top_k: 返回的文档数量
        filter_dict: 过滤条件，如 {"file_type": "pdf"}
        
    Returns:
        List[Document]: 相关文档列表
    """
    # 执行相似度搜索
    if filter_dict:
        docs = vector_store.similarity_search(query, k=top_k, filter=filter_dict)
    else:
        docs = vector_store.similarity_search(query, k=top_k)
    
    return docs


def retrieve_with_scores(
    query: str,
    vector_store,
    top_k: int = 5,
    filter_dict: Optional[Dict] = None
) -> List[tuple]:
    """
    检索相关文档并返回相似度分数
    
    Args:
        query: 查询文本
        vector_store: Chroma向量存储实例
        top_k: 返回的文档数量
        filter_dict: 过滤条件
        
    Returns:
        List[tuple]: (文档, 相似度分数) 列表，分数越小越相似
    """
    if filter_dict:
        docs_with_scores = vector_store.similarity_search_with_score(
            query, k=top_k, filter=filter_dict
        )
    else:
        docs_with_scores = vector_store.similarity_search_with_score(query, k=top_k)
    
    return docs_with_scores


def format_retrieved_docs(docs: List[Document]) -> str:
    """
    格式化检索结果用于LLM上下文
    
    Args:
        docs: 检索到的文档列表
        
    Returns:
        str: 格式化的上下文字符串
    """
    contexts = []
    
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', '未知来源')
        content = doc.page_content.strip()
        
        context = f"[文档{i}] 来自: {source}\n{content}\n"
        contexts.append(context)
    
    return "\n".join(contexts)


def create_retriever(vector_store, top_k: int = 5):
    """
    创建LangChain检索器对象
    
    Args:
        vector_store: Chroma向量存储实例
        top_k: 返回的文档数量
        
    Returns:
        Retriever: LangChain检索器
    """
    return vector_store.as_retriever(
        search_kwargs={"k": top_k}
    )



def generate_answer(
    question: str,
    context_docs: List[Document],
    llm,
    prompt_template: str = None
) -> str:
    """
    使用LLM生成答案 - 最简单常用的方式
    
    Args:
        question: 用户问题
        context_docs: 检索到的相关文档列表
        llm: 语言模型实例（从get_llm()获取）
        prompt_template: 自定义提示模板，None则使用默认模板
        
    Returns:
        str: 生成的答案
    """
    # 使用提示模板
    if prompt_template is None:
        prompt_template = DETAILED_TEMPLATE
    
    # 格式化上下文
    context = format_context(context_docs)
    
    # 创建提示
    prompt = ChatPromptTemplate.from_template(prompt_template)
    formatted_prompt = prompt.format(context=context, question=question)
    
    # 调用LLM生成答案
    response = llm.invoke(formatted_prompt)
    
    # 提取答案内容
    if hasattr(response, 'content'):
        return response.content
    else:
        return str(response)


def generate_answer_with_sources(
    question: str,
    context_docs: List[Document],
    llm
) -> Dict[str, Any]:
    """
    生成答案并返回来源信息
    
    Args:
        question: 用户问题
        context_docs: 检索到的相关文档
        llm: 语言模型实例
        
    Returns:
        Dict: 包含答案和来源的字典
    """
    # 使用提示模板
    prompt = ChatPromptTemplate.from_template(DETAILED_TEMPLATE)
    
    # 格式化上下文
    context = format_context_with_sources(context_docs)
    formatted_prompt = prompt.format(context=context, question=question)
    response = llm.invoke(formatted_prompt)
    
    answer = response.content if hasattr(response, 'content') else str(response)
    
    # 提取来源信息
    sources = []
    for doc in context_docs:
        source = doc.metadata.get('source', '未知来源')
        if source not in sources:
            sources.append(source)
    
    return {
        'answer': answer,
        'sources': sources,
        'document_count': len(context_docs)
    }


def format_context(docs: List[Document]) -> str:
    """
    简单格式化文档为上下文字符串
    
    Args:
        docs: 文档列表
        
    Returns:
        str: 格式化的上下文
    """
    contexts = []
    
    for i, doc in enumerate(docs, 1):
        content = doc.page_content.strip()
        contexts.append(f"[{i}] {content}")
    
    return "\n\n".join(contexts)


def format_context_with_sources(docs: List[Document]) -> str:
    """
    格式化文档为包含来源的上下文字符串
    
    Args:
        docs: 文档列表
        
    Returns:
        str: 格式化的上下文
    """
    contexts = []
    
    for i, doc in enumerate(docs, 1):
        content = doc.page_content.strip()
        source = doc.metadata.get('source', '未知来源')
        contexts.append(f"[{i}] 来自《{source}》:\n{content}")
    
    return "\n\n".join(contexts)