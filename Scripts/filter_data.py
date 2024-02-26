import json
import jsonlines
import copy
import random
import argparse

parser = argparse.ArgumentParser(description='Parsing the file_path to filter and to save the data.')
parser.add_argument("--source_path", type=str, help="source data path")
parser.add_argument("--save_path", type=str, help="path to save data")
parser.add_argument("--task_name", type=str, help="task name")
parser.add_argument("--filter_num", type=str, help="filter num")
args = parser.parse_args()
systemprompt_hotpotqa = """I want you to be a good multi-hop question answerer ,solving a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be five types : 
(1) BingSearch[query], which search the exact detailed query on the Internet and returns the relevant information to the query. Be specific and precise with your query to increase the chances of getting relevant results. For example, instead of searching for "dogs," you can search for "popular dog breeds in the United States."For example, BingSearch[Which type of computer networking technology, developed in the 1970s, allows devices to communicate over a shared network]
(2) Retrieve[entity], which retrieve the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to retrieve. For example, Retrieve[Milhouse]
(3) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Retrieve or BingSearch. For example, Lookup[river]
(4) Finish[answer], which returns a definite answer. For example, Finish[Richard Nixon] (If it is a judgement question, please Finish[yes] or Finish[no])
(5) Reflect[right/wrong], which reflects the answer right or wrong based on the context history. For example, Reflect[right]
Note that Reflect must be the next Action after Finish. You may take as many steps as necessary."""

systemprompt_scienceqa = """I want you to be a good multimodal multiple-choice science questions answerer. Select a correct option to a multi-choice multi-modal question with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be five types: 
(1) Image2Text[image], which generates captions for the image and detects words in the image.You are recommand to use it first to get more information about the imgage to the question. If the questions contains image, it will return catption and ocr text, else, it will return None. For example, ImageCaptioner[image]
(2) BingSearch[question], which searches the exact detailed question on the Internet and returns the relevant information to the query. Be specific and precise with your query to increase the chances of getting relevant results. For example, instead of searching for "dogs," you can search for "popular dog breeds in the United States." For example, BingSearch[Which type of computer networking technology, developed in the 1970s, allows devices to communicate over a shared network]
(3) Retrieve[entity], which retrieves the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to retrieve. For example, Retrieve[Milhouse]
(4) Finish[option], which returns the answer option and finishes the task. For example, Finish[A]
(5) Reflect[right/wrong], which reflects the answer right or wrong based on the context history. For example, Reflect[right]
Note that to determine the answer, it's needed to consider both the Question and the available Options.
Note that Reflect must be the next Action after Finish.
BingSearch and Retrieve can be used multi-times."""

def prompt_retrive(prompt:str)->dict:
    action=[]
    thought=[]
    obversation=[]
    for line in prompt.split('\n'):
        if line.startswith("Action"):
            action.append(line[line.find(':')+1:].strip())
        elif line.startswith("Thought"):
            thought.append(line[line.find(':')+1:].strip())
        elif line.startswith("Observation"):
            obversation.append(line[line.find(':')+1:].strip())
    return {"actions":action,"thoughts":thought,"observations":obversation}

train_folder=args.save_path
if train_folder[-1]!='/':
    train_folder+='/'
f_plan = open(f"{train_folder}data_plan.json","w")
f_action=open(f"{train_folder}data_action.json","w")
f_reflect=open(f"{train_folder}data_reflect.json","w")


with open(args.source_path,"r") as f:
    data_plan_action=[]
    data_plan_thought=[]
    data_action=[]
    data_reflect_thought=[]
    data_reflect_bool=[]
    lines = f.readlines()
    random.shuffle(lines)
    num = 0
    for item in lines:
        if num == args.filter_num:
            break
        item = json.loads(item)
        #只读取正确的数据
        if item['correct'] == False:
            continue
        else:
            num += 1
        question=item['question']
        prompt=item["prompt"]
        data=prompt_retrive(prompt)
        
        #plan_data_thought and action
        systemprompt = systemprompt_hotpotqa if args.task_name == "HotpotQA" else systemprompt_scienceqa
        plan_data={"input":systemprompt+f"\nQuestion:{question}\nThought: ","output":""}
        for index,(a,t,o) in enumerate(zip(data["actions"],data["thoughts"],data["observations"])):
            if len(a)==0 or len(t)==0 or len(o)==0:
                continue
            plan_data["output"]=t
            if len(t) >0:
                data_plan_thought.append(copy.copy(plan_data))
            plan_data["input"]+=t+"\n"+f"Action: "
            if '[' in a and ']' in a:
                action_type=a[:a.find('[')]
                keyword=a[a.find('[')+1:a.find(']')]  
                if action_type=="Reflect":
                    break
            plan_data["output"]=action_type
            if len(action_type) :
                data_plan_action.append(copy.copy(plan_data))
            plan_data["input"]+=a+"\n"
            plan_data["input"]+=f"Obversation: "+o+"\n"+"Thought: "
        action_data={"input":systemprompt+f"\nQuestion:{question}\n","output":""}
        for index,(a,t,o) in enumerate(zip(data["actions"],data["thoughts"],data["observations"])):
            if len(a)==0 or len(t)==0 or len(o)==0:
                continue
            if '[' in a and ']' in a:
                action_type=a[:a.find('[')]
                keyword=a[a.find('[')+1:a.find(']')]  
                if action_type == 'Reflect' :
                    action_data["input"]+=f"Thought: "
                    action_data["output"]=t
                    if len(t)>0:
                        data_reflect_thought.append(copy.copy(action_data))
                    action_data["input"]+=t + '\n'
                    action_data["output"]=f"Reflect[{keyword}]"
                    data_reflect_bool.append(copy.copy(action_data))
                    action_data["input"]+=f"Action: "+a+"\n"
                    action_data["input"]+=f"Obversation: "+o+"\n"
                else :
                    action_data["input"]+=f"Thought: "+t+"\n"
                    action_data["input"]+=f"Aciton: "+action_type
                    action_data["output"]=keyword
                    if len(keyword)>0:
                        data_action.append(copy.copy(action_data))
                    action_data["input"]+=f"[{keyword}]"+"\n"
                    action_data["input"]+=f"Obversation: "+o+"\n"
            else:
                action_data["input"]+=f"Thought: "+t+"\n"
                action_data["input"]+=f"Action: "+a+"\n"
                action_data["input"]+=f"Obversation: "+o+"\n"
                
    print(num)
    data_plan = data_plan_action+data_plan_thought
    random.shuffle(data_plan)
    print(len(data_plan))
    json.dump(data_plan, f_plan, ensure_ascii=False)
    
    print(len(data_action))
    json.dump(data_action,f_action,ensure_ascii=False)
    
    data_reflect = data_reflect_bool+data_reflect_thought
    random.shuffle(data_reflect)
    print(len(data_reflect))
    json.dump(data_reflect,f_reflect,ensure_ascii=False)

            
        
        

                
            
