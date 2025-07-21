import json

from langchain.agents import tool
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import LLMChain
from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import requests

import os

api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_PROXY")
serp_api_key = os.getenv("SERPAPI_API_KEY")

params = {
    "engine": "baidu",  # 更换搜索引擎为Bing
    "gl": "cn",        # 设置搜索国家为美国
    "hl": "zn",        # 设置搜索语言为英语
}

llm = ChatOpenAI(
    openai_api_key=api_key,
    openai_api_base=api_base,
    temperature=0,
)

@tool
def search(query:str):
    """只有需要了解实时信息或不知道的事情的时候才会使用这个工具。"""
    serp = SerpAPIWrapper(
        serpapi_api_key=serp_api_key,
        params=params
    )
    result = serp.run(query)
    print("实时搜索结果", result)
    return result

@tool
def get_info_from_local(query:str):
    """只有回答与2024运势或者龙年运势相关的问题的时候，才会使用这个工具"""
    client = Qdrant( #这里为什么要用client
        QdrantClient(path="E:/myAgent/local-qdrant"),
        "local_documents",
        OpenAIEmbeddings(
            openai_api_key=api_key,
            openai_api_base=api_base,
        ),
    )
    retriever = client.as_retriever(
        search_type="mmr"
    )

    result = retriever.get_relevant_documents(query)
    return result

@tool
def horoscope_calculation(query:str):
    """只有做八字排盘的时候才会使用到这个工具。需要输入用户姓名和出生年月日时，如果缺少用户姓名或出生年月日则工具不可用。"""
    url = f"https://api.yuanfenju.com/index.php/v1/Bazi/paipan"
    template = """你是一个参数查询助手，根据用户输入内容找出相关参数并按json格式返回
    JSON字段如下：-"api_key":"Jo9ygM2IEHdThGEqqTLdmqdO4", -"name":"姓名", -"sex":"性别",
    -"type":"日历类型，默认公历", -"year":"出生年份，例：1998", -"month":"出生月份，例：8", -"day":"出生日期，例：8", 
    -"hours":"出生小时，例：14", -"minute":"0"。如果没有相关参数，则需要提醒用户告诉你这些内容，只返回数据结构，不要有其他评论，
    否贼会有惩罚，用户输入：{query}
    """
    prompt = ChatPromptTemplate.from_template(template)

    parser = JsonOutputParser()
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())
    chain = prompt | llm | parser
    data = chain.invoke({"query":query})
    print("八字查询结果", data)
    result = requests.post(url, data=data)
    if result.status_code == 200:
        print(result.json())
        try:
            json = result.json()
            returnstring = f'八字为:{json["data"]["bazi_info"]["bazi"]}'
            return returnstring
        except Exception as e:
            return "八字查询失败，可能是你忘记询问用户姓名或者出生年月日时了。"
    else:
        return "技术错误，请稍后再试。"


@tool
def constellation_fortune(query:str):
    """只有用户想要测算星座运势的时候才会使用这个工具。"""
    constellations = {"白羊座":0, "金牛座":1, "双子座":2, "巨蟹座":3, "狮子座":4, "处女座":5,
                      "天秤座":6, "天蝎座":7, "射手座":8, "摩羯座":9, "水瓶座":10, "双鱼座":11}
    constellation = None

    url = f"https://api.yuanfenju.com/index.php/v1/Zhanbu/yunshi"

    template = """你是一个参数查询助手，根据用户输入内容找出相关参数并按json格式返回
    JSON字段如下：-"api_key":"Jo9ygM2IEHdThGEqqTLdmqdO4", -"type":"0", 
    -"title_yunshi":{title_yunshi}。如果没有相关参数，则需要提醒用户告诉你这些内容，只返回数据结构，不要有其他评论，
    否则会有惩罚，用户输入：{query}
    """
    prompt = ChatPromptTemplate.from_template(template)
    parser = JsonOutputParser()
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    chain = prompt | llm | parser

    for i, idx in constellations.items():
        if i in query:
            constellation = idx
            break

    data = chain.invoke({"query":query, "title_yunshi":constellation})
    print("运势数据为：", data)
    result = requests.post(url, data=data)
    if result.status_code == 200:
        print(result.json())
        try:
            json = result.json()
            returnstring = f'今日运势为:{json["data"]}'
            return returnstring
        except Exception as e:
            return "运势查询失败，可能是你忘记询问用户日期了。"
    else:
        return "技术错误，请稍后再试。"

@tool
def jiemeng(query:str):
    """只有用户需要解梦的时候才会用到这个工具，如果确实用户梦境内容则不可用。"""
    url = f"https://api.yuanfenju.com/index.php/v1/Gongju/zhougong"
    api_key = YUANFENJU_API_KEY
    template = """根据内容提取1个关键词，只返回关键词，
    内容为：{query}
    """
    prompt = PromptTemplate.from_template(template)

    chain = LLMChain(
        llm=llm,
        prompt=prompt,
    )

    keyword = chain.invoke({"query":query})
    print("关键词为：", keyword)
    result = requests.post(url, data={"api_key":api_key, "title_zhougong":keyword["text"]})
    if result.status_code == 200:
        print(result.json())
        return_string = json.loads(result.text)
        return return_string
    else:
        return "技术错误，请稍后再试"


