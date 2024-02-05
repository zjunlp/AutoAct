import json
import os
path = "test.json"
def save_to_json(data,path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, 'r') as file:
            ori_data = json.load(file)
    else:
        ori_data = []
    with open(path, 'w') as file:
        data = ori_data+data
        json.dump(data, file,indent=4)
for i in range(10):
    data=[{"a":i}]
    save_to_json(data,path)
for i in range(11,20):
    data=[{"a":i}]
    save_to_json(data,path)