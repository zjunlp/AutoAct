import torch
import argparse
import json
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import sys
sys.path.append('/data/rolnan/')
from pre_prompt import (
    ACTION_SYSTEM_PROMPT,
    BENCHMARK_DESCRIPTION,
    TOOL_POOL,
    TASK_PROMPT_TEMPLATE
)
from llms import MetaAgent


def action_parse(text):
    text = text.strip()
    lines = text.split("\n")
    actions = [line.strip() for line in lines if line != "" and line[0].isdigit()]
    return actions
    
def main(args):
    task_name = args.task_name
    task_description = BENCHMARK_DESCRIPTION[args.task_name]
    
    meta_agent = MetaAgent(
        model_name=args.model_name,
        openai_key=args.openai_key,
        url=args.openai_base,
        system_prompt=ACTION_SYSTEM_PROMPT
    )
    tool_pool = [{"name": tool["name"], "definition": tool["definition"]} for tool in TOOL_POOL]
    human_prompt_args = {"task_name": task_name, "task_description": task_description, "tool_pool": tool_pool}
    output = meta_agent.generate(
        human_prompt_template=TASK_PROMPT_TEMPLATE,
        human_prompt_args=human_prompt_args,
        temprature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        update_prompt=False
    )
    print(output)
    
    generated_actions = action_parse(output)
    tool_selected = []
    for action in generated_actions:
        for tool in TOOL_POOL:
            if tool["name"] in action:
                tool_selected.append(tool)
                break
    with open(args.tool_save_path, 'w') as json_file:
        json.dump(tool_selected, json_file, indent=2)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="/data/PLMs/llama-2-converted/7b-chat")
    parser.add_argument("--task_name", type=str, default="ScienceQA")
    parser.add_argument("--openai_key", type=str, default="EMPTY")  
    parser.add_argument("--openai_base", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.75)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--retrieve_k", type=int, default=3)
    parser.add_argument("--retrieve_p", type=float, default=0.6)
    parser.add_argument("--tool_save_path", type=str, default="/data/rolnan/ScienceQA/tool_selected.json")
    args = parser.parse_args()
    
    main(args)