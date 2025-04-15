# RAG助手
# from cy_tools import db_tools, model_tools
import pandas as pd
import os
from openai import OpenAI
from datetime import datetime
import re
import pprint
import urllib.parse
import json5
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool

import os

def get_file_list(folder_path):    
    # 初始化文件列表
    file_list = []
    
    # 遍历文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 获取文件的完整路径
            file_path = os.path.join(root, file)
            # 将文件路径添加到列表中
            file_list.append(file_path)
    return file_list

# 获取指定知识库文件列表 
file_list = get_file_list('./input')
#file_list = ['./制造\\【e-works】人工智能在制造业的应用现状调研报告【发现报告 fxbaogao.com】.pdf']

def get_application_bot(files, system_instruction, user_prompt):
    # 配置您所使用的 LLM。
    llm_cfg = {
        'model': 'qwen-max',
        'model_server': 'dashscope',
        'generate_cfg': {
            'top_p': 0.8
        }
    }
    
    tools = []    
    bot = Assistant(llm=llm_cfg,
                    system_message=system_instruction,
                    function_list=tools,
                    files=files)
    
    messages = []
    messages.append({'role': 'user', 'content': user_prompt})
    return bot.run_nonstream(messages=messages)

system_instruction = """你是大数据助手"""
user_prompt = "什么是业务过程？"

response = get_application_bot(file_list, system_instruction, user_prompt)
print(response[0]['content'])