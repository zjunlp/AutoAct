import os
import argparse 
import numpy as np
import pandas as pd
import concurrent
import joblib
from hotpotqa_run.utils import summarize_trial_detailed, log_trial
import hotpotqa_run.utils as utils
from hotpotqa_run.agent_arch import get_agent
from hotpotqa_run.llms import get_llm_backend
from hotpotqa_run.config import available_agent_names
import json

# os.environ["https_proxy"] = "http://127.0.0.1:7890"

parser = argparse.ArgumentParser(description='Parsing the input of agents, llms and llm context length.')
parser.add_argument("--agent_name", type=str, help="Name of the agent.", default="ZeroshotThink_HotPotQA_run_Agent")
parser.add_argument("--llm_name", type=str, help="Name of the llm", default="llama-2-13b-merge")
parser.add_argument("--max_context_len", type=int, help="Maximum context length", default=4096)
args = parser.parse_args()

agent_name = args.agent_name

llm_name=args.llm_name

max_context_len = args.max_context_len 

assert agent_name in available_agent_names

def process_agent_run_step(agent):
    agent.run()

def run_one_complex_level(level="easy"):
    hotpot = joblib.load(f'/data/rolnan/MetaBLOAA/hotpotqa_run/data/{level}.joblib').reset_index(drop = True)
    # hotpot = json.load(open("/data/rolnan/BOLAA/hotpotqa_run/data/llm_gen/metaqa_7b_11_26_final.json"))
    # agent_save_file = f"/data/rolnan/BOLAA/output/hotpotqa/train_data_7b_11_27.jsonl"
    agent_save_file = f"/data/rolnan/BOLAA/output/hotpotqa/13b-{level}-merge_12_23.jsonl"
    # task_instructions = [(row['Question'], row['Answer']) for row in hotpot]
    task_instructions = [(row['question'], row['answer']) for _, row in hotpot.iterrows()]
    # print(task_instructions)
    if os.path.exists(agent_save_file):
        sessions = utils.get_all_agent_sessions(agent_save_file)
        completed_tasks = utils.get_non_error_tasks(sessions)
        print(f"{level}:{len(completed_tasks)}")
        task_instructions = [task for task in task_instructions if task not in completed_tasks]
        utils.delete_error(agent_save_file)
    
    llm = get_llm_backend(llm_name).run
    
    agent_cls = get_agent(agent_name)
    agents = [agent_cls(ques, ans, llm, max_context_len) for ques, ans in task_instructions]
    # with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    #     executor.map(process_agent_run_step, agents)
    for agent in agents:
        process_agent_run_step(agent)
        utils.log_agent(agent, agent_save_file)        
    print(f'Finished Trial. Total: {len(agents)}')
    
def main():
    levels = ['easy', 'medium', 'hard']
    # levels = ['hard']
    for level in levels:
        run_one_complex_level(level)
    # run_one_complex_level()
if __name__ == '__main__':
    main()