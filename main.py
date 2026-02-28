import re
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_chroma import Chroma

from llms import get_llm, get_embeddings
from qa_chain import (
    process_question,
    retrieve_documents,
    format_retrieved_docs,
    format_context_with_sources,
    DETAILED_TEMPLATE
)



class RAGConfig:
    """RAG系统配置类"""
    
    def __init__(self):
        self.project_path = Path("C:/Users/Administrator/Desktop/llm_app/rag")
        self.data_dir = self.project_path / "data"
        self.chroma_dir = self.data_dir / "chroma"
        self.collection_name = "documents_collection"
        self.top_k = 5
        self.temperature = 0.7
        self.max_tokens = 4096


class RAGApplication:
    """RAG应用主类 - 整合所有组件"""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """
        初始化RAG应用
        
        Args:
            config: 配置对象，None则使用默认配置
        """
        self.config = config or RAGConfig()
        
        print("=" * 70)
        print("初始化RAG应用系统")
        print("=" * 70)
        
        # 初始化各组件
        self._init_llm()
        self._init_vector_store()
        self._init_chains()  
        self._init_tools()
        
        # 对话历史记录
        self.conversation_history = []
        
        print("\nRAG应用初始化完成")
        print("=" * 70)
        time.sleep(2)
    
    def _init_llm(self):
        """初始化语言模型"""
        print("\n1. 加载语言模型...")
        self.llm = get_llm()
        print(f"语言模型: {self.llm.model_name}")
    
    def _init_vector_store(self):
        """初始化向量数据库"""
        print("\n2. 加载向量数据库...")
        embeddings = get_embeddings()
        self.vector_store = Chroma(
            persist_directory=str(self.config.chroma_dir),
            embedding_function=embeddings,
            collection_name=self.config.collection_name
        )
        print(f"向量数据库加载成功")
        print(f"存储目录: {self.config.chroma_dir}")
    
    def _init_chains(self):
        """构建完整的Chain执行链 - 不使用有问题的导入"""
        print("\n3. 构建Chain执行链...")
        
        # 创建检索器
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.config.top_k}
        )
        
        # 创建提示模板
        self.prompt = ChatPromptTemplate.from_template(DETAILED_TEMPLATE)
        
        # 方法1: 使用LCEL构建链（推荐）- 不需要额外的导入
        self.lcel_chain = (
            RunnableParallel({
                "context": self.retriever | self._format_docs_for_chain,
                "question": RunnablePassthrough()
            })
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        # 方法2: 自定义检索链 - 替代 create_retrieval_chain
        self.simple_chain = self._create_simple_retrieval_chain()
        
        print(f"链构建完成")
        print(f"检索top_k: {self.config.top_k}")
        print(f"链类型: LCEL + 自定义检索链")
    
    def _format_docs_for_chain(self, docs):
        """为链格式化文档"""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def _create_simple_retrieval_chain(self):
        """
        创建简单的检索链 - 替代 create_retrieval_chain
        """
        def run_chain(inputs):
            question = inputs.get("input", "")
            
            # 检索文档
            docs = self.retriever.get_relevant_documents(question)
            
            # 格式化上下文
            context = self._format_docs_for_chain(docs)
            
            # 生成提示
            prompt_value = self.prompt.invoke({
                "context": context,
                "question": question
            })
            
            # 调用LLM
            response = self.llm.invoke(prompt_value)
            
            # 返回结果
            return {
                "input": question,
                "context": docs,
                "answer": response.content if hasattr(response, 'content') else str(response)
            }
        
        return run_chain
    
    def _init_tools(self):
        """初始化工具集"""
        print("\n4. 加载工具集...")
        
        self.tools = {
            "calculator": self._calculator,
            "datetime": self._get_datetime,
            "extract_numbers": self._extract_numbers,
            "count_words": self._count_words
        }
        
        print(f"已加载 {len(self.tools)} 个工具:")
        for tool_name in self.tools.keys():
            print(f"--{tool_name}")
    

    
    def _calculator(self, expression: str) -> str:
        """
        简单计算器工具
        
        Args:
            expression: 数学表达式，如 "1+2*3"
            
        Returns:
            str: 计算结果
        """
        try:
            # 安全评估数学表达式
            allowed_chars = set("0123456789+-*/(). ")
            if not all(c in allowed_chars for c in expression):
                return "错误：表达式包含非法字符"
            
            # 使用安全的eval
            result = eval(expression, {"__builtins__": {}}, {})
            return f"计算结果: {expression} = {result}"
        except Exception as e:
            return f"计算错误: {e}"
    
    def _get_datetime(self, format: str = "%Y-%m-%d %H:%M:%S") -> str:
        """
        获取当前日期时间
        
        Args:
            format: 日期时间格式
            
        Returns:
            str: 格式化的日期时间
        """
        return datetime.now().strftime(format)
    
    def _extract_numbers(self, text: str) -> str:
        """
        从文本中提取所有数字
        
        Args:
            text: 输入文本
            
        Returns:
            str: 提取的数字列表
        """
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            return f"提取到的数字: {', '.join(numbers)}"
        else:
            return "未找到数字"
    
    def _count_words(self, text: str) -> str:
        """
        统计文本词数
        
        Args:
            text: 输入文本
            
        Returns:
            str: 词数统计
        """
        words = re.findall(r'[\u4e00-\u9fa5a-zA-Z]+', text)
        return f"词数统计: {len(words)} 个词"
    
    def call_tool(self, tool_name: str, *args, **kwargs) -> str:
        """
        调用工具
        
        Args:
            tool_name: 工具名称
            *args, **kwargs: 工具参数
            
        Returns:
            str: 工具执行结果
        """
        if tool_name not in self.tools:
            return f"错误: 未知工具 '{tool_name}'"
        
        try:
            return self.tools[tool_name](*args, **kwargs)
        except Exception as e:
            return f"工具调用失败: {e}"
    
    
    def answer_with_lcel(self, question: str) -> str:
        """
        使用LCEL链回答问题
        
        Args:
            question: 用户问题
            
        Returns:
            str: 生成的答案
        """
        try:
            processed_q = process_question(question)
            answer = self.lcel_chain.invoke(processed_q)
            return answer
        except Exception as e:
            return f"生成答案失败: {e}"
    
    def answer_with_simple_chain(self, question: str) -> str:
        """
        使用简单检索链回答问题
        
        Args:
            question: 用户问题
            
        Returns:
            str: 生成的答案
        """
        try:
            processed_q = process_question(question)
            result = self.simple_chain({"input": processed_q})
            return result['answer']
        except Exception as e:
            return f"生成答案失败: {e}"
    
    def answer_with_details(self, question: str) -> Dict[str, Any]:
        """
        回答问题并返回详细信息
        
        Args:
            question: 用户问题
            
        Returns:
            Dict: 包含答案、来源、检索文档等详细信息
        """
        start_time = time.time()
        
        # 处理问题
        processed_q = process_question(question)
        
        # 检索文档
        docs = retrieve_documents(processed_q, self.vector_store, top_k=self.config.top_k)
        
        # 准备结果
        result = {
            "question": question,
            "processed_question": processed_q,
            "answer": "",
            "sources": [],
            "document_count": len(docs),
            "retrieval_time": 0,
            "generation_time": 0,
            "total_time": 0
        }
        
        if not docs:
            result["answer"] = "未找到相关文档，无法回答问题。"
            result["total_time"] = time.time() - start_time
            return result
        
        # 记录检索时间
        retrieval_time = time.time()
        result["retrieval_time"] = retrieval_time - start_time
        
        # 生成答案
        context = format_context_with_sources(docs)
        prompt = ChatPromptTemplate.from_template(DETAILED_TEMPLATE)
        formatted_prompt = prompt.format(context=context, question=question)
        
        response = self.llm.invoke(formatted_prompt)
        answer = response.content if hasattr(response, 'content') else str(response)
        
        # 记录生成时间
        generation_time = time.time()
        result["generation_time"] = generation_time - retrieval_time
        result["total_time"] = generation_time - start_time
        
        # 提取来源
        sources = []
        for doc in docs:
            source = doc.metadata.get('source', '未知来源')
            if source not in sources:
                sources.append(source)
        
        result["answer"] = answer
        result["sources"] = sources
        
        # 保存到历史记录
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "sources": sources,
            "total_time": result["total_time"]
        })
        
        return result
    
    
    def answer_with_tools(self, question: str) -> str:
        """
        集成工具的问答（检测是否需要调用工具）
        
        Args:
            question: 用户问题
            
        Returns:
            str: 生成的答案
        """
        # 检测是否需要调用工具
        if "计算" in question and any(op in question for op in ["+", "-", "*", "/"]):
            # 提取数学表达式
            numbers = re.findall(r'[\d+\-*/().]+', question)
            if numbers:
                expr = numbers[0]
                return self.call_tool("calculator", expr)
        
        elif "时间" in question or "日期" in question:
            return self.call_tool("datetime")
        
        elif "提取数字" in question:
            return self.call_tool("extract_numbers", question)
        
        elif "词数" in question or "单词数" in question:
            return self.call_tool("count_words", question)
        
        # 默认使用RAG问答
        return self.answer_with_lcel(question)

    
    def print_welcome(self):
        """打印欢迎信息"""
        print("\n" + "=" * 70)
        print("欢迎使用RAG智能问答系统")
        print("=" * 70)
        print("可用命令:")
        print("  /help    - 显示帮助信息")
        print("  /tools   - 显示可用工具")
        print("  /history - 显示对话历史")
        print("  /stats   - 显示系统统计")
        print("  /clear   - 清屏")
        print("  /exit    - 退出系统")
        print("=" * 70)
    
    def print_help(self):
        """打印帮助信息"""
        print("\n" + "-" * 50)
        print("帮助信息")
        print("-" * 50)
        print("1. 直接输入问题即可获得答案")
        print("2. 系统会自动检索相关文档并生成答案")
        print("3. 工具调用示例:")
        print("   - 计算: '计算 123+456'")
        print("   - 时间: '现在几点了？'")
        print("   - 提取数字: '从文本中提取数字'")
        print("   - 词数统计: '统计这段文字的词数'")
        print("-" * 50)
    
    def print_tools(self):
        """打印可用工具"""
        print("\n" + "-" * 50)
        print("可用工具")
        print("-" * 50)
        for tool_name in self.tools.keys():
            print(f"  - {tool_name}")
        print("-" * 50)
    
    def print_history(self, limit: int = 5):
        """打印对话历史"""
        print("\n" + "-" * 50)
        print(f"最近对话历史 (共{len(self.conversation_history)}条)")
        print("-" * 50)
        
        if not self.conversation_history:
            print("暂无对话历史")
        else:
            for i, record in enumerate(self.conversation_history[-limit:], 1):
                print(f"{i}. Q: {record['question']}")
                print(f"   A: {record['answer'][:50]}...")
        print("-" * 50)
    
    def print_stats(self):
        """打印系统统计"""
        print("\n" + "-" * 50)
        print("系统统计")
        print("-" * 50)
        print(f"总对话次数: {len(self.conversation_history)}")
        print(f"检索top_k: {self.config.top_k}")
        print(f"向量数据库: {self.config.chroma_dir}")
        print(f"语言模型: {self.llm.model_name}")
        
        # 计算平均响应时间
        if self.conversation_history:
            total_time = sum(record.get('total_time', 0) for record in self.conversation_history)
            avg_time = total_time / len(self.conversation_history)
            print(f"平均响应时间: {avg_time:.2f}秒")
        print("-" * 50)
    
    def clear_screen(self):
        """清屏"""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
        self.print_welcome()
    
    def run_cli(self):
        """运行命令行交互界面"""
        self.print_welcome()
        
        while True:
            try:
                # 获取用户输入
                user_input = input("\n📝 请输入问题: ").strip()
                
                # 处理命令
                if user_input.lower() == '/exit':
                    print("👋 感谢使用，再见！")
                    break
                elif user_input.lower() == '/help':
                    self.print_help()
                    continue
                elif user_input.lower() == '/tools':
                    self.print_tools()
                    continue
                elif user_input.lower() == '/history':
                    self.print_history()
                    continue
                elif user_input.lower() == '/stats':
                    self.print_stats()
                    continue
                elif user_input.lower() == '/clear':
                    self.clear_screen()
                    continue
                elif not user_input:
                    continue
                
                # 显示思考中
                print("🤔 思考中...")
                
                # 调用问答
                result = self.answer_with_details(user_input)
                
                # 显示结果
                print("\n" + "=" * 70)
                print(f"问题: {result['question']}")
                print("-" * 70)
                print(f"答案: {result['answer']}")
                
                if result['sources']:
                    print("-" * 70)
                    print("信息来源:")
                    for i, source in enumerate(result['sources'], 1):
                        print(f"  {i}. {source}")
                
                print("-" * 70)
                print(f"处理时间: {result['total_time']:.2f}秒 "
                      f"(检索: {result['retrieval_time']:.2f}秒, "
                      f"生成: {result['generation_time']:.2f}秒)")
                print("=" * 70)
                
            except KeyboardInterrupt:
                print("\n\n👋 检测到中断，退出系统")
                break
            except Exception as e:
                print(f"❌ 发生错误: {e}")
                time.sleep(1)
    
    def run_single_query(self, question: str):
        """
        运行单次查询（非交互模式）
        
        Args:
            question: 问题文本
        """
        result = self.answer_with_details(question)
        
        print("\n" + "=" * 70)
        print(f"问题: {result['question']}")
        print("=" * 70)
        print(f"答案: {result['answer']}")
        
        if result['sources']:
            print("\n信息来源:")
            for i, source in enumerate(result['sources'], 1):
                print(f"  {i}. {source}")
        
        print(f"\n处理时间: {result['total_time']:.2f}秒")
        print("=" * 70)



def main():
    """主程序入口"""
    import sys
    
    # 创建应用实例
    app = RAGApplication()
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        # 如果有命令行参数，作为问题处理
        question = ' '.join(sys.argv[1:])
        app.run_single_query(question)
        time.sleep(10)
    else:
        # 否则启动交互式CLI
        app.run_cli()


if __name__ == "__main__":
    main()