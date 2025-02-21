# CyberSercurity AI Chat Bot v0.0.1版本开发文档

## 主程序运行大纲:

### 一、程序功能概述
该程序主要实现基于文档的问答系统功能，通过读取文档、建立索引、处理用户问题并生成回答，同时支持调试日志记录和对话历史上下文筛选。

### 二、依赖导入
导入标准库：os、sys、json、toml、datetime。
导入自定义模块：DocumentReader、FaissIndexer、Embedding、ResponseGenerator、ContextFilter。
导入OpenAI相关模块。

### 三、配置文件读取
从config/parameter.toml读取 Open - AI 相关配置参数（如api_base、api_key、model_name等）和 BERT 模型名称。

### 四、Open - AI 初始化
使用读取的配置信息初始化OpenAI客户端。

### 五、日志相关操作
确保logs目录存在，若不存在则创建。提供保存和加载上次使用文档名的函数。

### 六、主程序逻辑
（一）模式判断
根据命令行参数判断是否为调试模式。
（二）文档路径获取
尝试加载上次使用的文档名。
根据用户选择决定是否使用上次文档，否则提示用户输入新的文档路径，并保存此次使用的文档名。
（三）文档处理
使用DocumentReader读取文档内容。
将文档内容按段落分割。
（四）索引建立
使用Embedding和FaissIndexer建立文档段落的 Faiss 索引。
（五）初始化
初始化对话历史列表。
初始化回答计数和日志记录计数。
初始化上下文筛选器和回答生成器。
（六）问答循环
读取用户输入，若为q则退出循环。
生成用户输入的向量并搜索索引获取相关上下文。
使用相关上下文和对话历史生成回答。
打印回答并更新对话历史。
每回答 10 次筛选一次对话历史上下文。
把筛选后文本赋值给历史记录,作为新的回复生成参考。
若为调试模式，记录调试日志。

## 其他模块运行大纲

### config
采用toml格式的配置文件,语法简洁，便于统一导入`api_key,api_base,model_name,以及temperature和top_k基本参数`

### docx
采用txt纯文本格式,本模型兼容docs,pdf格式文件,用于构建faiss索引,增强回复效果。

###  reader
#### reader.py
程序大纲:
总体逻辑为:
使用`def_readfile`读取文件,根据后缀名使用类中对应函数进行读取
采用`@staticmethod`装饰器,不需要实例化即可直接使用该类中的函数,

### generator
#### embedding.py
程序大纲:
引入关键的`transformer`库,分别使用`AutoTokenizer`和`AutoModel`类定义分词器和预处理器,获取可以给faiss库使用处理的嵌入向量。

#### response_generator.py
程序大纲:
接收 openai 模型名、温度、核采样阈值和 openaiI 客户端实例，用于配置回复生成的参数和连接 openai 服务。
整合对话历史和相关上下文，构造用于提问的提示词。
调用 OpenAI 的聊天完成接口，传入提示词和配置参数，获取回复。
根据是否需要返回提示词，返回回复内容或回复内容与提示词。

### filter
#### context_filter
程序大纲:
首先将上下文处理为换行符连接形式,使用openai库对其进行筛选处理后,再采用分割的方法,处理筛选后的上下文,最后使用列表赋值的方式将处理过后的重要上下文赋值给important_context,返回重要文本,(即还是列表中存在元组的形式),即筛选后文本

## 程序运行说明:

### 安装相关依赖

在项目根目录下运行:

```powershell
pip install -f requirements.txt
```

且需要ollama本地模型启动,确保网络流畅,以部署openai api

### 启动程序

```powershell
python main.py #普通运行模式

python main.py -d #调试模式 
```

