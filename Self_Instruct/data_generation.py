import argparse
from pre_prompt import (
    DATA_GEN_SYSTEM_PROMPT,
    HOTPOTQA_TASK_NAME,
    HOTPOTQA_TASK_DESCRIPTION,
    SCIENCEQA_TASK_NAME,
    SCIENCEQA_TASK_DESCRIPTION,
    SCIENCEQA_DATA_GEN_HUMAN_PROMPT,
    HOTPOTQA_DATA_GEN_HUMAN_PROMPT
)
from llms import MetaAgent
import json
import random
import re
import os
def get_data_hotpotqa(source_data):
    data=json.load(open(source_data))
    data=[{"Question":d['Question'],"Answer":d['Answer']} for d in data]
    return data

def get_data_scienceqa(source_data):
    data=json.load(open(source_data))
    data=[{"Question":d['Question'],"Answer":d['Answer'],"Caption":d["caption"]} for d in data]
    return data

def get_random_data(data, num_samples=5):
    random_data = random.sample(data, num_samples)
    return random_data

def parse_ouput_scienceqa(output):
    output = output.split("\n")
    non_empty_lines = [line for line in output if line.strip()]  
    output="\n".join(non_empty_lines)
    # Define a regular expression pattern to extract relevant information
    pattern = re.compile(r"Question:(.*?)Options:(.*?)Ocr:(.*?)Caption:(.*?)Answer:(.*?)$", re.DOTALL | re.MULTILINE)

    # Find all matches in the text
    matches = pattern.findall(output)

    # Create a list to store extracted data
    questions_data = []

    # Process each match
    for match in matches:
        question = match[0].strip()
        options = match[1].strip()
        ocr = match[2].strip()
        caption = match[3].strip()
        answer = match[4].strip()
        # Create a dictionary for each question
        question_data = {
            "Question": question,
            "Options": options,
            "Ocr": ocr,
            "Caption": caption,
            "Answer": answer
        }
        # Add the dictionary to the list
        print(len(questions_data),'\n')
        questions_data.append(question_data)


    return questions_data

def parse_ouput_hotpotqa(output):
    pattern = r"Question: (.+)\nAnswer: (.+)"
    matches = re.findall(pattern, output)
    print(matches)
    new_qa_pairs=[]
    for match in matches:
        question = match[0]
        answer = match[1]
        new_qa_pairs.append({
            'Question': question,
            'Answer': answer,
        })
    return new_qa_pairs
def save_to_json(data,path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, 'r') as file:
            ori_data = json.load(file)
    else:
        ori_data = []
    with open(path, 'w') as file:
        data = data+ori_data
        json.dump(data, file,indent=4)
        
def main(args):
    data_system_prmpt = DATA_GEN_SYSTEM_PROMPT
    if args.dataset_name == "hotpotqa":
        dataset_system_prompt = data_system_prmpt.format(task_name = HOTPOTQA_TASK_NAME, task_description = HOTPOTQA_TASK_DESCRIPTION)
    elif args.dataset_name == "scienceqa":
        dataset_system_prompt = data_system_prmpt.format(task_name = SCIENCEQA_TASK_NAME, task_description = SCIENCEQA_TASK_DESCRIPTION)
    meta_agent = MetaAgent(
        model_name=args.model_name,
        openai_key=args.openai_key,
        url=args.openai_base,
        system_prompt= dataset_system_prompt 
    )
    qa_pairs=[]
    if args.dataset_name=="hotpotqa":
        qa_pairs=get_data_hotpotqa(args.source_data)
    elif args.dataset_name == "scienceqa":
        qa_pairs = get_data_scienceqa(args.source_data)
    #else dataset_name
    answer_set=set()
    unique_qa=[]
    ori_qa=get_random_data(qa_pairs,num_samples=2)
    unique_qa = ori_qa
    for u in unique_qa:
        if args.dataset_name == "hotpotqa":
            answer_set.add(u["Answer"])
        elif args.dataset_name == "scienceqa":
            answer_set.add(u["Answer"].split(' ')[1])
    while(len(answer_set)<args.generate_all_num):
        print(f"have generated num {len(answer_set)}, all {args.generate_all_num} need to be generated all")
        all_qa=""
        sample_pairs=get_random_data(ori_qa,num_samples=2)+get_random_data(unique_qa,num_samples=min(3,len(unique_qa)))
        random.shuffle(sample_pairs)


        for qa in sample_pairs:
            all_qa+=str(qa)[1:-1]+"\n"
        human_prompt_args = {"QA_pairs":all_qa,"Gen_num":args.generate_per_round_num}
        human_prompt_template = HOTPOTQA_DATA_GEN_HUMAN_PROMPT if args.dataset_name == "hotpotqa" else SCIENCEQA_DATA_GEN_HUMAN_PROMPT
        output = meta_agent.generate(
            human_prompt_template=human_prompt_template,
            human_prompt_args=human_prompt_args,
            temprature=random.uniform(0.1, 0.5),
            top_k=args.top_k,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            update_prompt=False
        )
        print(output)
        data_list = []
        if(args.dataset_name == "hotpotqa"):
            for new_pair in parse_ouput_hotpotqa(output):
                print(new_pair,'\n')
                if new_pair["Answer"] not in answer_set and len(new_pair["Answer"])<20:
                    unique_qa.append(new_pair)
                    answer_set.add(new_pair["Answer"])
                    data_list.append(new_pair)
            save_to_json(data_list,args.target_data)
        else:
            for new_pair in parse_ouput_scienceqa(output):
                print(new_pair,'\n')
                if new_pair["Answer"].split(' ')[1] not in answer_set :
                    unique_qa.append(new_pair)
                    answer_set.add(new_pair["Answer"].split(' ')[1])
                    data_list.append(new_pair)
            save_to_json(data_list,args.target_data)
            

            
        
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="/data/PLMs/llama-2-converted/7b-chat")
    parser.add_argument("--source_data", type=str, default="/data/rolnan/hotpotqa/test.json")
    parser.add_argument("--generate_all_num", type=int, default=10)
    parser.add_argument("--generate_per_round_num", type=int, default=10)
    parser.add_argument("--target_data",type=str,default="/data/rolnan/hotpotqa/generate_test.json")
    parser.add_argument("--dataset_name", type=str, default="scienceqa")
    parser.add_argument("--openai_key", type=str, default="EMPTY")
    parser.add_argument("--openai_base", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.75)
    parser.add_argument("--max_tokens", type=int, default=1024)
    args = parser.parse_args()
    
    main(args)