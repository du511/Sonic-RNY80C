# Sonic RNY80C

本项目是一个基于RAG文档的问答系统，结合了信息检索和自然语言处理技术，能够根据用户输入的问题，利用预加载的文档信息生成相关回答。系统支持从不同格式的文档（如 `.txt`、`.pdf`、`.docx`）中提取信息，并通过 Faiss 索引进行高效的文档段落检索。同时，利用sklearn库使用朴素贝叶斯进行问题类型判断,结合文档进行针对性回答。
根据本地ai名称修改`config/parameter.toml`文件中模型名,默认:`rabbit`

## 使用

1. 安装依赖项

```
pip install -r requirements.txt
```

2. 下载你要使用的RAG文档,存放在docs目录下. 运行项目时请输入具体文件名以及其路径

```
python main.py
```
调试模式:

```
python main.py -d
```
会生成日志

4. 输入问题，系统会自动生成回答

## 开发文档

另见`CyberSercurity AI Chat Bot v0.0.1版本开发文档.md`

## 致谢

ollama项目以及requirements.txt中的所以依赖项作者

## 许可证

Apache License 2.0
