import os
import argparse 
import numpy as np
import pandas as pd
import concurrent
import joblib
from benchmark_run.utils import summarize_trial_detailed, log_trial
import benchmark_run.utils as utils
from benchmark_run.Meta_agent_arch import get_agent
from benchmark_run.llms import get_llm_backend
from benchmark_run.config import available_agent_names
import json

# os.environ["https_proxy"] = "http://127.0.0.1:7890"
S_model = "13"
T_model = "70"
ft_num=300
parser = argparse.ArgumentParser(description='Parsing the input of agents, llms and llm context length.')
parser.add_argument("--agent_name", type=str, help="Name of the agent.", default="ZeroshotThink_HotPotQA_run_Agent")
parser.add_argument("--plan_agent", type=str, help="Name of the plan", default=f"{S_model}b_from_{T_model}b_{ft_num}_plan_peft")
parser.add_argument("--action_agent", type=str, help="Name of the action", default=f"{S_model}b_from_{T_model}b_{ft_num}_action_peft")
parser.add_argument("--reflect_agent", type=str, help="Name of the reflect_agent", default=f"{S_model}b_from_{T_model}b_{ft_num}_reflect_peft")
parser.add_argument("--max_context_len", type=int, help="Maximum context length", default=4096)
parser.add_argument("--task",type=str ,help="task name",default="Hotpotqa")
parser.add_argument("--task_path",type=str,help="task path")
parser.add_argument("--save_path",type=str,help="save path")
args = parser.parse_args()

agent_name = args.agent_name

plan_agent = args.plan_agent
action_agent = args.action_agent
reflect_agent = args.reflect_agent
max_context_len = args.max_context_len 
save_path = args.save_path
if save_path[-1] != "/":
    save_path += "/"
task_path = args.task_path
assert agent_name in available_agent_names

def process_agent_run_step(agent):
    agent.run()

def run_one_complex_level_hotpotqa(level="easy"):
    hotpot = joblib.load(f'{task_path}/{level}.joblib').reset_index(drop = True)
    agent_save_file = f"{save_path}{level}.jsonl"
    task_instructions = [(row['question'], row['answer']) for _, row in hotpot.iterrows()]
    if os.path.exists(agent_save_file):
        sessions = utils.get_all_agent_sessions(agent_save_file)
        completed_tasks = utils.get_non_error_tasks(sessions)
        print(f"{level}:{len(completed_tasks)}")
        task_instructions = [task for task in task_instructions if task not in completed_tasks]
        utils.delete_error(agent_save_file)
    
    llm_plan = get_llm_backend(plan_agent).run
    llm_action = get_llm_backend(action_agent).run
    llm_reflect = get_llm_backend(reflect_agent).run
    
    agent_cls = get_agent(agent_name)
    agents = [agent_cls(ques, ans, llm_plan, llm_action, llm_reflect, max_context_len) for ques, ans in task_instructions]
    for agent in agents:
        process_agent_run_step(agent)
        utils.log_agent(agent, agent_save_file)        
    print(f'Finished Trial. Total: {len(agents)}')
def run_one_complex_level_scienceqa(level="1-4"):
    f = open(f'{task_path}/format_scienceqa_grade{level}.json')
    scienceqa = json.load(f)
    agent_save_file = f"{save_path}{level}.jsonl"
    task_instructions = [(row['Question'],row['choices'],row['Answer'],row['orc'],row['caption']) for row in scienceqa]
    if os.path.exists(agent_save_file):
        sessions = utils.get_all_agent_sessions(agent_save_file)
        completed_tasks = utils.get_non_error_tasks(sessions)
        print(f"{level}:{len(completed_tasks)}")
        task_instructions = [task for task in task_instructions if task[0] not in completed_tasks]
        utils.delete_error(agent_save_file)
    
    llm_plan = get_llm_backend(plan_agent).run
    llm_action = get_llm_backend(action_agent).run
    llm_reflect = get_llm_backend(reflect_agent).run
    
    agent_cls = get_agent(agent_name)
    agents = [agent_cls(ques, choices, ans, caption, orc, llm_plan, llm_action, llm_reflect, max_context_len) for ques,choices,ans,orc,caption in task_instructions]
    for agent in agents:
        process_agent_run_step(agent)
        utils.log_agent(agent, agent_save_file)        
    print(f'Finished Trial. Total: {len(agents)}')
    
def main():
    if args.task == "Hotpotqa":
        levels = ['easy', 'medium', 'hard']
        for level in levels:
            run_one_complex_level_hotpotqa(level)
    elif args.task == "Scienceqa":
        levels = ['1-4', '5-8', '9-12']
        for level in levels:
            run_one_complex_level_scienceqa(level)
if __name__ == '__main__':
    main()