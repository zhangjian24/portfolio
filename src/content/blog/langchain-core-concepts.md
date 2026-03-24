---
title: "LangChain核心概念解析"
description: "2026年最新版LangChain核心概念详解：Model I/O、Chain、Memory、LCEL、LangGraph（含执行流程）"
pubDate: 2026-03-23
updatedDate: "2026-03-23"
heroImage: '/image/logo.svg'
tags: ["LangChain", "LLM", "教程", "入门"]
---

# LangChain 核心概念解析

> 2026年最新版 | Model I/O、Chain、Memory、LCEL 全面掌握

## 一、引言

LangChain 是由 Harrison Chase 于2022年10月发起的开源LLM应用开发框架，比ChatGPT问世还要早一个月。经过近三年的发展，LangChain已从单一开源包成长为覆盖开发、调试、部署全流程的完整生态系统。

截至2026年，LangChain在GitHub上已获得超过90,000 Stars，每月数百万次下载，成为企业级LLM应用开发的标准基础设施。本文将深入解析LangChain的四大核心概念：Model I/O、Chain、Memory、LCEL，帮助你建立完整的知识体系。

---

## 二、整体架构概览

### 2.1 分层架构设计

LangChain采用分层架构，不同层级各司其职又可协同工作：

```
┌─────────────────────────────────────────────────────────────┐
│                      LangChain 生态                         │
├─────────────────────────────────────────────────────────────┤
│  应用层 │ langgraph (Agent编排、状态管理、工作流)            │
├─────────────────────────────────────────────────────────────┤
│  组件层 │ langchain (Chains、Agents、Retrieval)             │
├─────────────────────────────────────────────────────────────┤
│  集成层 │ langchain-community (第三方模型、工具、向量库)     │
├─────────────────────────────────────────────────────────────┤
│  核心层 │ langchain-core (基础抽象、LCEL、Runnable接口)     │
└─────────────────────────────────────────────────────────────┘
```

- **langchain-core**：提供基础抽象与LCEL，是组件协同的核心
- **langchain-community**：第三方集成模块，覆盖Model I/O、Retrieval、Tool等
- **langchain**：包含Chains、Agents、Retrieval等核心业务组件
- **langgraph**：编排多个节点，负责整个工作流的调度与状态跳转

### 2.2 核心组件关系

```
用户输入 → Prompt Templates → LLM → Output Parsers → 最终输出
                    ↓              ↓
               Memory ←─────── Chain ←─────── Tools
```

---

## 三、Model I/O：与LLM的沟通桥梁

Model I/O是应用与LLM交互的核心模块，类似于JDBC与数据库的关系。其核心价值在于**解耦应用逻辑与底层模型实现**，让你可以自由切换不同提供商（OpenAI、Anthropic、Google等）而不改变业务代码。

### 3.1 三步工作流程

Model I/O的工作流程分为三个关键步骤：

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    Format   │ →  │   Predict   │ →  │    Parse    │
│  (输入格式化) │    │  (模型调用)  │    │  (输出解析)  │
└─────────────┘    └─────────────┘    └─────────────┘
     ↓                  ↓                  ↓
 Prompt Templates    Chat Models      Output Parsers
```

### 3.2 Prompt Templates

Prompt Templates 用于结构化提示词，支持变量替换和复用：

```python
from langchain_core.prompts import ChatPromptTemplate

# 方式一：from_messages（推荐）
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}，专门帮助用户解决{topic}问题"),
    ("human", "我的问题是：{question}")
])

# 方式二：from_template（传统方式）
prompt = ChatPromptTemplate.from_template(
    "你是{role}，请用{style}风格回答以下问题：{question}"
)

# 格式化输入
formatted_prompt = prompt.format(
    role="技术顾问",
    topic="编程",
    question="Python如何处理异常？",
    style="简洁专业"
)

print(formatted_prompt)
```

**输出**：
```
System: 你是一个技术顾问，专门帮助用户解决编程问题
Human: 我的问题是：Python如何处理异常？
```

### 3.3 Chat Models

LangChain提供统一的模型接口，支持多种提供商：

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# OpenAI
openai_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    streaming=True  # 启用流式输出
)

# Anthropic
claude_llm = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0.7
)

# Google Gemini
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.7
)
```

### 3.4 Output Parsers

Output Parsers 将原始输出转换为结构化数据：

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel

# 字符串解析器（最常用）
str_parser = StrOutputParser()

# JSON解析器
json_parser = JsonOutputParser()

# Pydantic解析器（推荐用于结构化输出）
class Answer(BaseModel):
    result: str
    confidence: float
    sources: list[str]

pydantic_parser = PydanticOutputParser(pydantic_object=Answer)

# 使用示例
response = llm.invoke("What is 2+2?")
parsed = str_parser.invoke(response)
print(parsed)  # "2"
```

### 3.5 完整调用示例

将三个组件串联成完整流程：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 初始化模型
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 创建Prompt模板
prompt = ChatPromptTemplate.from_template(
    "用一句话解释{concept}，用中文回答"
)

# 创建输出解析器
parser = StrOutputParser()

# 方式一：传统方式（手动调用）
formatted_prompt = prompt.format(concept="LLM")
response = llm.invoke(formatted_prompt)
result = parser.invoke(response)
print(result)

# 方式二：LCEL管道方式（推荐）
chain = prompt | llm | parser
result = chain.invoke({"concept": "LLM"})
print(result)
```

---

## 四、Chains：串联组件的工作流

Chains（链）用于将多个组件串联成自动化工作流，实现步骤化任务处理。

### 4.1 LLMChain

最基本的链，用于将Prompt模板、LLM、输出解析器组合：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

llm = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_template(
    "帮我写一首关于{topic}的诗"
)

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(topic="春天")
print(result)
```

### 4.2 Sequential Chain

顺序执行多个链：

```python
from langchain.chains import SequentialChain
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate

# 链1：生成标题
title_chain = LLMChain(
    llm=llm,
    prompt=ChatPromptTemplate.from_template("为以下内容生成一个标题：{content}"),
    output_key="title"
)

# 链2：生成摘要
summary_chain = LLMChain(
    llm=llm,
    prompt=ChatPromptTemplate.from_template("用50字概括以下内容：{content}"),
    output_key="summary"
)

# 组合顺序链
sequential_chain = SequentialChain(
    chains=[title_chain, summary_chain],
    input_variables=["content"],
    output_variables=["title", "summary"]
)

result = sequential_chain.invoke({"content": "Python是一门易学难精的编程语言..."})
print(result)
```

### 4.3 传统方式 vs LCEL方式对比

| 特性 | 传统方式 (LLMChain) | LCEL方式 |
|------|---------------------|----------|
| 语法 | 面向对象，方法调用 | 管道运算符 `\|` |
| 灵活性 | 较低，固定模式 | 高，可自由组合 |
| 并行支持 | 需额外处理 | 原生支持 |
| 流式输出 | 需额外配置 | 原生支持 |
| 异步支持 | 需异步封装 | 原生async/await |

---

## 五、LCEL语法：LangChain Expression Language

LCEL是LangChain最强大的特性之一，它将所有组件统一为Runnable接口，通过管道运算符实现声明式组合。

### 5.1 Runnable接口

LangChain v1.x中，所有组件都实现了`Runnable`接口：

```python
from langchain_core.runnables import Runnable

# Runnable接口的核心方法
class Runnable:
    def invoke(self, input, config=None):
        """同步调用"""
        pass

    async def ainvoke(self, input, config=None):
        """异步调用"""
        pass

    def stream(self, input, config=None):
        """流式输出"""
        pass

    def batch(self, inputs, config=None):
        """批量处理"""
        pass
```

### 5.2 管道运算符

`|` 运算符将前一个组件的输出作为下一个组件的输入：

```python
# 基本语法
chain = component_a | component_b | component_c

# 等价于
result = component_c.invoke(component_b.invoke(component_a.invoke(input)))
```

### 5.3 完整示例对比

**传统方式**：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_template("用中文介绍{topic}")
parser = StrOutputParser()

# 手动串联
formatted = prompt.format(topic="人工智能")
response = llm.invoke(formatted)
result = parser.invoke(response)
print(result)
```

**LCEL方式**：

```python
# 一行搞定
chain = prompt | llm | parser
result = chain.invoke({"topic": "人工智能"})
print(result)
```

### 5.4 并行与分支

LCEL支持并行处理和条件分支：

```python
from langchain_core.runnables import RunnableParallel, RunnableBranch

# 并行执行多个分支
parallel_chain = RunnableParallel(
    chinese = prompt | llm | parser,
    english = (prompt | llm | parser).bind(language="en"),
    code = code_prompt | llm | parser
)

# 条件分支
branch_chain = RunnableBranch(
    (lambda x: x.get("type") == "code", code_chain),
    (lambda x: x.get("type") == "data", data_chain),
    default_chain  # 默认链
)
```

### 5.5 流式输出

LCEL原生支持流式输出：

```python
chain = prompt | llm | parser

# 流式输出（一个字一个字显示）
for chunk in chain.stream({"topic": "量子计算"}):
    print(chunk, end="", flush=True)
```

### 5.6 LCEL的优势

| 优势 | 说明 |
|------|------|
| **简洁性** | 用管道运算符替代嵌套函数调用 |
| **可组合性** | 组件可自由组合，易于扩展 |
| **原生支持** | 并行、流式、异步开箱即用 |
| **可观测性** | 易于调试和监控 |
| **性能** | 官方称并行处理效率提升30%-50% |

---

## 六、Memory：对话状态的守护者

Memory（内存）用于在对话过程中维护状态，解决LLM上下文窗口有限的问题。

### 6.1 短期记忆 vs 长期记忆

```
┌─────────────────────────────────────────────────────────────┐
│                      Memory 分类                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   短期记忆                    长期记忆                      │
│   ┌──────────┐              ┌──────────────┐              │
│   │ 单次会话 │              │ 跨会话持久化  │              │
│   │ 上下文保持│              │ 知识积累      │              │
│   └──────────┘              └──────────────┘              │
│        ↓                          ↓                         │
│   BufferMemory              VectorStore                    │
│   BufferWindowMemory        RetrieverMemory                │
│   SummaryMemory                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 ConversationBufferMemory

最简单的内存，保存完整对话历史：

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini")

# 创建带记忆的链
memory = ConversationBufferMemory(
    memory_key="history",
    return_messages=True  # 返回消息对象列表
)

prompt = ChatPromptTemplate.from_template(
    """基于以下对话历史回答问题：

历史：
{history}

问题：{question}"""
)

chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# 多轮对话
print(chain.invoke({"question": "我叫张三"}))
print(chain.invoke({"question": "我叫什么名字？"}))
```

### 6.3 ConversationBufferWindowMemory

限制保存最近N轮对话，避免历史过长：

```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(
    k=3,  # 只保留最近3轮对话
    memory_key="chat_history",
    return_messages=True
)
```

### 6.4 ConversationSummaryMemory

自动总结对话要点，适合长对话：

```python
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(
    llm=llm,  # 用于生成摘要的LLM
    memory_key="summary",
    return_messages=True
)
```

### 6.5 VectorStoreRetrieverMemory：基于向量检索的语义记忆

`VectorStoreRetrieverMemory` 是LangChain提供的长期记忆组件，它将对话内容存储在向量数据库中，通过语义相似性检索来回忆相关信息。与短期记忆不同，它**不显式跟踪对话顺序**，而是根据语义相关性动态检索最"显著"的记忆片段。

#### 6.5.1 核心特点

| 特性 | 说明 |
|------|------|
| **存储方式** | 向量数据库（FAISS、Chroma、Pinecone等） |
| **检索方式** | 语义相似性搜索，而非时间顺序 |
| **适用场景** | 跨会话记忆、语义检索、长周期知识积累 |
| **优势** | 支持海量记忆、按语义检索、不受token限制 |

#### 6.5.2 工作原理

```
┌─────────────────────────────────────────────────────────────────┐
│              VectorStoreRetrieverMemory 工作流程                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  保存记忆：                                                      │
│  输入 → 嵌入模型 → 向量存储                                      │
│                                                                 │
│  检索记忆：                                                      │
│  查询 → 嵌入模型 → 向量相似度搜索 → 返回Top-K结果                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 6.5.3 初始化向量存储

**方式一：FAISS（本地向量库）**

```python
import faiss
from langchain.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 初始化嵌入模型和LLM
embedding = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4o-mini")

# 创建FAISS向量存储
embedding_size = 1536  # OpenAIEmbeddings维度
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(index, embedding, InMemoryDocStore(), {})

# 创建retriever（k=3表示返回最相关的3条记忆）
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 创建向量记忆
memory = VectorStoreRetrieverMemory(
    retriever=retriever,
    memory_key="chat_history"
)

# 保存一些对话记忆
memory.save_context(
    {"input": "我最喜欢的食物是披萨"},
    {"output": "好的，我记住了 你喜欢披萨"}
)
memory.save_context(
    {"input": "我喜欢在周末去跑步"},
    {"output": "运动是个很好的习惯！"}
)
memory.save_context(
    {"input": "我最近在学习Python编程"},
    {"output": "Python是一门很实用的语言，继续加油！"}
)

# 检索相关记忆
print(memory.load_memory_variables({"prompt": "我应该吃什么？"}))
```

**输出**：
```
{'chat_history': 'input: 我最喜欢的食物是披萨\noutput: 好的，我记住了...'}
```

**方式二：Chroma（本地向量库）**

```python
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# 创建Chroma向量存储
vectorstore = Chroma.from_documents(
    documents=[],  # 初始为空
    embedding=OpenAIEmbeddings()
)

# 创建memory
memory = VectorStoreRetrieverMemory(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    memory_key="history"
)
```

**方式三：Pinecone（云端向量数据库）**

```python
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import VectorStoreRetrieverMemory

# 连接Pinecone
vectorstore = PineconeVectorStore.from_params(
    index_name="my-index",
    embedding=OpenAIEmbeddings(),
    namespace="memory"
)

memory = VectorStoreRetrieverMemory(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    memory_key="long_term_memory"
)
```

#### 6.5.4 在ConversationChain中使用

将向量记忆与对话链结合：

```python
# 创建对话链
prompt = ChatPromptTemplate.from_template("""
基于以下记忆回答用户问题。如果没有相关信息，请如实说明。

记忆：
{chat_history}

问题：{input}
""")

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=True
)

# 第一次对话
response1 = conversation.invoke({"input": "我的爱好是什么？"})
print(response1)

# 第二次对话（跨会话）
response2 = conversation.invoke({"input": "我应该学习什么编程语言？"})
print(response2)
```

#### 6.5.5 高级配置：检索策略

**相似度搜索（默认）**

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",  # 默认
    search_kwargs={"k": 3}
)
```

**最大边际相关性（MMR）- 更具多样性**

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",  # 避免返回相似度过高的结果
    search_kwargs={"k": 5, "fetch_k": 20}
)
```

**带分数阈值的检索**

```python
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5, "score_threshold": 0.7}
)
```

#### 6.5.6 使用场景与注意事项

| 适用场景 | 不适用场景 |
|----------|------------|
| 跨会话持久化记忆 | 需要严格时间顺序的场景 |
| 基于语义而非关键词的检索 | 短对话、简单上下文 |
| 海量历史信息存储 | 实时性要求极高的场景 |
| 个性化用户画像构建 | 隐私敏感的数据 |

**注意事项**：

1. **k值选择**：k越大，检索越多，但可能引入噪声；k越小，可能遗漏重要信息
2. **嵌入模型**：选择与场景匹配的嵌入模型，中文推荐text-embedding-3-small或中文专用模型
3. **存储成本**：云端向量数据库有存储费用，本地FAISS/Chroma免费但占用磁盘
4. **定期清理**：长期使用后需要清理无效记忆，避免检索质量下降

### 6.6 LangGraph中的持久化

LangGraph提供检查点机制实现状态持久化：

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START

# 创建带持久化的图
checkpointer = MemorySaver()

graph = StateGraph(GraphState)
graph.add_node("process", process_node)
graph.add_edge(START, "process")
graph.compile(checkpointer=checkpointer)

# 线程ID用于恢复状态
config = {"configurable": {"thread_id": "user_123"}}
result = graph.invoke({"input": "hello"}, config)

# 恢复状态继续执行
result = graph.invoke({"input": "continue"}, config)
```

### 6.7 LangGraph 执行流程

LangGraph 的执行流程清晰简洁，核心围绕**状态（State）**、**节点（Node）**、**边（Edge）**三个概念展开：

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph 执行流程                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐             │
│   │  State  │ ←── │  Node   │ ←── │  Edge   │             │
│   │ (状态)  │     │ (节点)  │     │  (边)   │             │
│   └─────────┘     └─────────┘     └─────────┘             │
│       ↑                                         │         │
│       └─────────────────────────────────────────┘         │
│                    循环执行直到完成                         │
└─────────────────────────────────────────────────────────────┘
```

**核心概念**：

| 概念 | 说明 | 关键函数 |
|------|------|----------|
| State | 图的共享状态，可以是dict或Pydantic模型 | `StateGraph(GraphState)` |
| Node | 执行逻辑的函数，接收state并返回更新 | `graph.add_node(name, func)` |
| Edge | 定义节点间的流转关系 | `graph.add_edge()` / `graph.add_conditional_edges()` |
| compile | 将图编译为可执行的app | `graph.compile()` |

**最小示例**：

```python
from langgraph.graph import StateGraph, START, END

# 1. 定义状态类型
class GraphState(TypedDict):
    input: str
    result: str

# 2. 定义节点函数
def node_a(state: GraphState) -> GraphState:
    return {"result": f"处理: {state['input']}"}

def node_b(state: GraphState) -> GraphState:
    return {"result": state["result"] + " + 节点B"}

# 3. 构建图
graph = StateGraph(GraphState)
graph.add_node("A", node_a)
graph.add_node("B", node_b)
graph.add_edge(START, "A")    # 起点 → A
graph.add_edge("A", "B")      # A → B
graph.add_edge("B", END)      # B → 终点

# 4. 编译并执行
app = graph.compile()
result = app.invoke({"input": "hello"})
print(result)  # {'input': 'hello', 'result': '处理: hello + 节点B'}
```

**执行流程**：
1. 用户调用 `app.invoke(input)` 传入初始状态
2. 图从 START 节点开始，按照边的定义依次执行
3. 每个节点接收当前state，处理后返回更新后的state
4. 执行到 END 节点或无出边时结束
5. 返回最终状态作为结果

与Chain相比，LangGraph适合需要**多步骤状态传递**、**条件分支**、**人机协作**的复杂场景，是构建AI Agent的核心框架。

---

## 七、总结与进阶路径

### 7.1 核心概念回顾

| 概念 | 作用 | 关键类 |
|------|------|--------|
| Model I/O | 与LLM交互的标准化接口 | ChatOpenAI, PromptTemplate, OutputParser |
| Chain | 串联组件的工作流 | LLMChain, SequentialChain |
| LCEL | 声明式组件组合语法 | 管道运算符 `\|` |
| Memory | 对话状态管理 | BufferMemory, VectorStoreMemory |

### 7.2 2026年新特性

1. **LCEL全面取代旧语法**：v1.x推荐全部使用LCEL
2. **LangGraph整合**：复杂工作流优先使用LangGraph
3. **MCP协议支持**：标准化工具和数据接入
4. **LangSmith监控**：生产环境必备

### 7.3 进阶学习路径

```
LangChain基础
    ↓
LCEL深入 + RAG实战
    ↓
LangGraph工作流
    ↓
Agent开发 + 多Agent系统
```

---

## 参考资料

- [LangChain官方文档](https://python.langchain.com/docs/introduction/)
- [LangChain v1.x迁移指南](https://python.langchain.com/docs/versions/migrating/)
- [LCEL最佳实践](https://python.langchain.com/docs/expression_language/)
- [LangGraph文档](https://langchain-ai.github.io/langgraph/)
