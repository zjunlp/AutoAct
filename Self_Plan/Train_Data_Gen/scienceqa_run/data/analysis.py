import json
import random


# grade1_4, grade5_8, grade9_12 = [], [], []
# with open("/data/rolnan/BOLAA/scienceqa_run/data/format_scienceqa.json") as f:
#     datas = json.load(f)
#     for data in datas:
#         grade = data["metadata"]["grade"]
#         if 1 <= grade <= 4:
#             grade1_4.append(data)
#         elif 5 <= grade <= 8:
#             grade5_8.append(data)
#         elif 9 <= grade <= 12:
#             grade9_12.append(data)
            
# print(len(grade1_4), len(grade5_8), len(grade9_12))

# random.shuffle(grade1_4)
# random.shuffle(grade5_8)
# random.shuffle(grade9_12)

# with open("/data/rolnan/BOLAA/scienceqa_run/data/format_scienceqa_grade1-4.json", "w") as f:
#     json.dump(grade1_4[:120], f, ensure_ascii=False)
    
# with open("/data/rolnan/BOLAA/scienceqa_run/data/format_scienceqa_grade5-8.json", "w") as f:
#     json.dump(grade5_8[:120], f, ensure_ascii=False)
    
# with open("/data/rolnan/BOLAA/scienceqa_run/data/format_scienceqa_grade9-12.json", "w") as f:
#     json.dump(grade9_12[:120], f, ensure_ascii=False)

pids_1_4, pids_5_8, pids_9_12 = [], [], []
with open("/data/rolnan/BOLAA/scienceqa_run/data/format_scienceqa_grade1-4.json") as f:
    datas = json.load(f)
    for data in datas:
        pids_1_4.append(data["pid"])
        
with open("/data/rolnan/BOLAA/scienceqa_run/data/format_scienceqa_grade5-8.json") as f:
    datas = json.load(f)
    for data in datas:
        pids_5_8.append(data["pid"])
        
with open("/data/rolnan/BOLAA/scienceqa_run/data/format_scienceqa_grade9-12.json") as f:
    datas = json.load(f)
    for data in datas:
        pids_9_12.append(data["pid"])

pids = {}   
with open("/data/rolnan/BOLAA/scienceqa_run/data/test_pids.json", "w") as f:
    pids_1_4.sort()
    pids_5_8.sort()
    pids_9_12.sort()
    pids["test_easy"] = pids_1_4
    pids["test_medium"] = pids_5_8
    pids["test_hard"] = pids_9_12
    json.dump(pids, f)