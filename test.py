import json

# 指定 JSONL 文件路径
jsonl_file_path = '/data/rolnan/AutoAct/Self_Plan/Tool_Selection/HotpotQA_Tools.json'

# 打开文件并逐行读取
string = ''
with open(jsonl_file_path, 'r') as file:
    tools = json.load(file)
    for index,tool in enumerate(tools):
        string += f"{index+1}. {tool['usage']}\n"
print(string)
