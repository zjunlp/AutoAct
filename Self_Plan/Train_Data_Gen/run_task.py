import os
import argparse 
import numpy as np
import pandas as pd
import concurrent
import joblib
from benchmark_run.utils import summarize_trial_detailed, log_trial
import benchmark_run.utils as utils
from benchmark_run.agent_arch import get_agent
from benchmark_run.llms import get_llm_backend
from benchmark_run.config import available_agent_names
import json

# os.environ["https_proxy"] = "http://127.0.0.1:7890"

parser = argparse.ArgumentParser(description='Parsing the input of agents, llms and llm context length.')
parser.add_argument("--agent_name", type=str, help="Name of the agent.", default="ZeroshotThink_HotPotQA_run_Agent")
parser.add_argument("--llm_name", type=str, help="Name of the llm", default="llama-2-13b-merge")
parser.add_argument("--max_context_len", type=int, help="Maximum context length", default=4096)
parser.add_argument("--task",type=str ,help="task name",default="Hotpotqa")
parser.add_argument("--task_path",type=str,help="task path")
parser.add_argument("--save_path",type=str,help="save path")
args = parser.parse_args()

agent_name = args.agent_name

llm_name=args.llm_name
task_path=args.task_path
save_path=args.save_path
max_context_len = args.max_context_len 

assert agent_name in available_agent_names

def process_agent_run_step(agent):
    agent.run()

def run_one_complex_level_hotpotqa():
    hotpot = json.load(open(task_path))
    agent_save_file = save_path
    task_instructions = [(row['Question'], row['Answer']) for row in hotpot]
    if os.path.exists(agent_save_file):
        sessions = utils.get_all_agent_sessions(agent_save_file)
        completed_tasks = utils.get_non_error_tasks(sessions)
        task_instructions = [task for task in task_instructions if task not in completed_tasks]
        utils.delete_error(agent_save_file)
    llm = get_llm_backend(llm_name).run
    
    agent_cls = get_agent(agent_name)
    agents = [agent_cls(ques, ans, llm, max_context_len) for ques, ans in task_instructions]
    for agent in agents:
        process_agent_run_step(agent)
        utils.log_agent(agent, agent_save_file)        
    print(f'Finished Trial. Total: {len(agents)}')

def run_one_complex_level_scienceqa():
    scienceqa = json.load(open(task_path))
    agent_save_file = save_path
    task_instructions = [(row['Question'], row["choices"],row['Answer'], row["caption"], row["orc"]) for row in scienceqa]
    if os.path.exists(agent_save_file):
        sessions = utils.get_all_agent_sessions(agent_save_file)
        completed_tasks = utils.get_non_error_tasks(sessions)
        # task_instructions = [task for task in task_instructions if task not in completed_tasks]
        task_instructions = task_instructions[len(completed_tasks):]
        utils.delete_error(agent_save_file)
    llm = get_llm_backend(llm_name).run
    agent_cls = get_agent(agent_name)
    agents = [agent_cls(ques, ans, llm, choices, cap, ocr, max_context_len) for ques, choices, ans, cap, ocr in task_instructions]
    for agent in agents:
        process_agent_run_step(agent)
        utils.log_agent(agent, agent_save_file)
    print(f'Finished Trial. Total: {len(agents)}')

def main():
    if args.task == "Hotpotqa":
        run_one_complex_level_hotpotqa()
    elif args.task == "Scienceqa":
        run_one_complex_level_scienceqa()
if __name__ == '__main__':
    main()