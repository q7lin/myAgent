# 服务端端口 -> langchain -> openai
# 客户端：电报机器人、微信机器人、website
# 接口：http，https，websocket

# 服务器：
1. 接口访问，python选型fastapi
2. /chat的接口，post请求
3. /add_urls 从url中学习知识
4. /add_pdfs 从pdf里学习知识
5. /add_texts 从txt文本里学习

# 人性化
1. 用户输入 -> AI判断以下当前问题的情绪倾向 -> 判断 -> 反馈 -> agent判断
2. 工具调用：用户发起请求->agent判断使用哪个工具->带着相关的参数去请求工具->得到观察结果

# 验证redis是否正在运行：
- 在终端中运行以下命令以连接到redis服务器:
    '''
    redis-cli
    '''
- 输入'ping'命令，如果返回'PONG'，则标识redis已成功安装并正在运行

### 截至目前
1. api
2. agent框架 #哪些代码算agent框架
3. tools：搜索、查询信息、专业知识库
4. 记忆、长时记忆
5. 学习能力

## 从url来学习，实现增强
1. 输入url
2. 地址的HTML变成可学习的文本
3. 向量化
4. 检索->相关文本块
5. LLM回答