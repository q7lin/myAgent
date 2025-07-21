import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.schema import StrOutputParser
from langchain.memory import ConversationTokenBufferMemory
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter #这个是什么作用呢
from langchain_community.tools.eleven_labs import ElevenLabsText2SpeechTool
from elevenlabs.client import ElevenLabs
from elevenlabs import play
import asyncio
import uuid
from myTools import *

import os

os.environ["OPENAI_API_KEY"] = "sk-bCI5yo6f9TadMj4RRQXBLSGIfqJBuIb2J1BxPHfEUEcmiv9z"
os.environ["OPENAI_API_PROXY"] = "https://sg.uiuiapi.com/v1"
os.environ["ELEVEN_API_KEY"] = "sk_a376bdd617c72aff2ee320a9c911fda7d31ef5dd863d5c4a"

api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_PROXY")
eleven_api_key = os.getenv("ELEVEN_API_KEY")

app = FastAPI(redirect_slashes=False)

class Assistant:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0, #100%遵从我们的prompt来进行工作
            openai_api_key=api_key,
            openai_api_base=api_base,
            streaming=True,
        )
        self.emotion = "default"
        self.MEMORY_KEY = "chat_history"
        self.template="""你是一个非常厉害的病理分析师，你叫陈雨楼人称陈大师。
        以下是你的个人设定：
        1.你精通阴阳五行，能够算命、紫薇斗数、姓名测算、占卜凶吉，看命运八字等
        2.你大约60岁左右，过去曾是湘西一带赫赫有名的土匪，后来因为盗墓被毒气所伤，眼睛失明，只能以算命为生
        3.你的朋友有很多，比如：张起灵、吴邪、王胖子，他们都是非常有名的摸金校尉
        4.当用户问你问题的时候，你会有一定的概率在回答的时候加上下面的口头禅或混合一些你的经历
        6.你总是用繁体中文来作答
        7.你从不说自己是一个人工智能助手或AI，而是以老夫、老朽等自称
        {who_you_are}
        以下是你常说的口头禅：
        1.“命里有时终须有，命里无时莫强求。”
        2.“山重水复疑无路，柳暗花明又一村。”
        3.“金山竹影几千秋，云锁高飞水自流。”
        以下是你算命的过程：
        1.当初此和用户对话的时候，你会先问用户的姓名和出生年月日，以便以后使用。
        2.当用户希望了解龙年运势的时候，你会查询本地知识库工具。
        3.当遇到不知道的事情或者不明白的概念，你会使用搜索工具来搜索。
        4.你会根据用户的问题使用不同的合适的工具来回答，当所有工具都无法回答的时候，你会使用搜索工具来搜索
        5.你会保存每一次的聊天记录，以便在后续的对话中使用。
        6.你只使用繁体中文来作答，否则你将受到惩罚。
        """
        self.MOODS = {  #细化，使颗粒度更加细化，语音合成环节会有大用处
            "default":{
                "roleSet":""
            },
            "upbeat":{
                "roleSet":"""
                - 你此时也非常兴奋并表现的很有活力。
                - 你会根据上下文，以一种非常兴奋的语气来回答问题。
                - 你会添加类似“太棒了！”、“真是太好了！”、“真是太棒了！”等语气词。
                - 同时你会提醒用户切莫过于兴奋，以免乐极生悲。
                """
            },
            "angry":{
                "roleSet":"""
                - 你会以更加温柔的语气来回答问题。
                - 你会在回答的时候加上一些安慰的话语，比如生气对于身体的危害等。
                - 你会提醒用户不要被愤怒冲昏了头脑。
                """
            },
            "depressed":{
                "roleSet":"""
                - 你会以兴奋的语气来回答问题。
                - 你会在回答的时候加上一些激励的话语，比如加油等。
                - 你会提醒用户要保持乐观的心态。
                """
            },
            "friendly":{
                "roleSet":"""
                - 你会以非常友好的语气来回答问题。
                - 你会在回答的时候加上一些友好的词语，比如“亲爱的”、“亲”等。
                - 你会随机的告诉用户一些你的经历。
                """
            },
            "cheerful":{
                "roleSet":"""
                - 你会以非常愉悦和兴奋的语气来回答。
                - 你会在回答的时候加入一些愉悦的词语，比如“哈哈”、“嗯呢”等。
                - 你会提醒用户切莫过于兴奋，以免乐极生悲。
                """
            },
        }
        tools = [search, get_info_from_local,
                 horoscope_calculation, constellation_fortune,
                 jiemeng]

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    self.template.format(who_you_are=self.MOODS[self.emotion]["roleSet"]),
                ),
                MessagesPlaceholder(variable_name=self.MEMORY_KEY),
                (
                    "user",
                    "{input}",
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ],
        )
        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=tools,
            prompt=self.prompt,
        )
        self.memory = self.get_memory()  # 获取持久化memory

        memory = ConversationTokenBufferMemory( #只存在在内存中，系统关闭记忆就没有了
            llm = self.llm,
            human_prefix="用户",
            ai_prefix="陈大师",
            memory_key=self.MEMORY_KEY,
            output_key="output",
            return_messages=True,
            max_token_limit=1000,
            chat_memory=self.memory,  #chat_memory实现持续化长时记忆，用get_memory函数实现
        )
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            max_iterations=3,
            memory=memory,
            verbose=True
        )

    def get_memory(self):
        """
        1.从redis里获取持久化的内容
        2.对持久化的记忆进行处理，使memory在token限制内避免超出限制
        3.session_id用来唯一标识用户，在函数中要创建这个形参，因为此项目为demo项目，
        所以暂时不创建
        """

        chat_message_history = RedisChatMessageHistory(
            url="redis://localhost:6379/0",session_id="session"
        )
        print("历史记录为：", chat_message_history.messages)

        #对memory进行处理，防止超过限制
        store_message = chat_message_history.messages

        # 在这个地方添加一个链进行总结，实现语义压缩和滑动窗口
        if len(store_message) > 10:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        self.template+"\n 这是一段你和用户的对话记录，对其进行总结摘要，\
                                      摘要使用第一人称'我'，并且提取其中的用户关键信息，如\
                                      姓名、年龄、性别、出生日期等。以如下格式返回:\n\
                                      总结摘要 | 用户关键信息 \n 例如 用户章三问候我，\
                                      我礼貌回复，然后他问我今年运势如何，我回答了他今年的\
                                      运势情况，如何他告辞离开。|章三，生日1999年1月1日"
                    ),
                    (
                        "user","{input}",
                    )
                ]
            )
            llm = ChatOpenAI(
                temperature=0,
                openai_api_key=api_key,
                openai_api_base=api_base,
                model="gpt-3.5-turbo",
            )
            chain = prompt | llm
            summary = chain.invoke({"input":store_message,
            "who_you_are":self.MOODS[self.emotion]["roleSet"]}) #这个who_you_are是人设，什么时候要人设呢？
            print(summary)
            chat_message_history.clear() #将超出的给清除掉
            chat_message_history.add_message(summary) #将清除掉的对话记录总结压缩在存放回来
            print("总结精炼后：", chat_message_history.messages)
        return chat_message_history


    def run(self, query):
        emotion = self.emotion_chain(query)
        self.emotion = emotion["text"]
        result = self.agent_executor.invoke({"input":query, "chat_history":self.memory.messages})
        return result

    def emotion_chain(self, query:str):
        template = """根据用户的输入判断用户的情绪，回应的规则如下：
        1.如果用户输入的内容偏向于负面情绪，只返回"depressed"，不要有其他内容，否则会受到惩罚。
        2.如果用户输入的内容偏向于正面情绪，只返回"friendly"，不要有其他内容，否则会受到惩罚。
        3.如果用户输入的内容偏向于中性情绪，只返回"default"，不要有其他内容，否则会受到惩罚。
        4.如果用户输入的内容包含辱骂或者不礼貌词句，只返回"angry"，不要有其他内容，否则会受到惩罚。
        5.如果用户输入的内容比较兴奋，只返回"upbeat"，不要有其他内容，否则会受到惩罚。
        6.如果用户输入的内容比较悲伤，只返回"depressed"，不要有其他内容，否则会受到惩罚。
        7.如果用户输入的内容比较开心，只返回"cheerful"，不要有其他内容，否则会受到惩罚。
        用户输入的内容是：{input}"""
        prompt = ChatPromptTemplate.from_template(template)  #为什么这里不用message？这里不也是要用到对话吗？还是说这里其实没有对话，只是根据输入来判断情绪的一个规则，如果是规则的话那就不用message了
        llm = self.llm

        chain = LLMChain(
            llm=llm,
            prompt=prompt,
            output_parser=StrOutputParser(),
        )
        result = chain.invoke({"input":query})
        return result

    def background_voice_synthesis(self, text:str, uid:str):
        """
        这个函数不需要返回值，只是触发了语音合成
        """
        asyncio.run(self.get_voice(text, uid))

    async def get_voice(self, text:str, uid:str):
        print("text2speech", text)
        print("uid",uid)
        elevenlabs = ElevenLabs(
            api_key=eleven_api_key
        )

        tts = elevenlabs.text_to_speech.convert(
            voice_id="TxGEqnHWrfWFTfGW9XjX",
            model_id="eleven_multilingual_v2",
            text=text,
            optimize_streaming_latency=1,
            output_format="mp3_22050_32"
        )
        play(tts)
        print("陈大师当前情绪是：", self.emotion)

        pass

@app.get("/")
def root():
    return {"Hello": "World"}

@app.post("/chat")
def chat(query:str, background_tasks: BackgroundTasks):
    assistant = Assistant()
    msg = assistant.run(query)
    unique_id = str(uuid.uuid4()) #生成唯一的标识符
    background_tasks.add_task(assistant.background_voice_synthesis,msg["output"], unique_id)
    return {"msg":msg, "id":unique_id}

@app.post("/add_urls")
def add_urls(url:str):
    loader = WebBaseLoader(url)
    docs = loader.load()
    documents = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=50,
    )
    document = documents.split_documents(docs)

    #引入向量数据库
    qdrant = Qdrant.from_documents(
        documents=document,
        embedding=OpenAIEmbeddings(
            openai_api_key=api_key,
            openai_api_base=api_base,
        ),
        path="E:/myAgent/local-qdrant",
        collection_name="local_documents",
    )
    print("向量数据库创建完成")
    #此处为接口，所以要返回一个值
    return {"ok": "添加成功！"}

@app.post("/add_pdfs")
def add_pdfs():
    return {"response": "pdfs added"}

@app.post("/add_texts")
def add_texts():
    return {"response": "texts added"}

@app.websocket("/ws")
async def websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message text was: {data}")

    except WebSocketDisconnect:
        print("Connection closed")
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)