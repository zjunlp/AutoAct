import os
import argparse 
import numpy as np
import pandas as pd
from scienceqa_run.utils import summarize_trial_detailed, log_trial
import scienceqa_run.utils as utils
from scienceqa_run.agent_arch import get_agent
from scienceqa_run.llms import get_llm_backend
from scienceqa_run.config import available_agent_names
import json

parser = argparse.ArgumentParser(description='Parsing the input of agents, llms and llm context length.')
parser.add_argument("--agent_name", type=str, help="Name of the agent.", default="ZeroshotThink_ScienceQA_run_Agent")
parser.add_argument("--llm_name", type=str, help="Name of the llm", default="llama-2-70b-chat")
parser.add_argument("--max_context_len", type=int, help="Maximum context length", default=4096)
args = parser.parse_args()

agent_name = args.agent_name
llm_name = args.llm_name
max_context_len = args.max_context_len
assert agent_name in available_agent_names

def process_agent_run_step(agent):
    agent.run()



def run_one_complex_level(level="easy"):
    scienceqa = json.load(open(f"/data/rolnan/BOLAA/scienceqa_run/data/format_scienceqa_grade{level}.json"))
    agent_save_file = f"/data/rolnan/BOLAA/output/scienceqa/70b-{level}-wo-ft.jsonl"
    task_instructions = [(row['Question'], row["choices"],row['Answer'], row["caption"], row["orc"]) for row in scienceqa]
    if os.path.exists(agent_save_file):
        sessions = utils.get_all_agent_sessions(agent_save_file)
        completed_tasks = utils.get_non_error_tasks(sessions)
        print(f"{level}:{len(completed_tasks)}")
        # task_instructions = [task for task in task_instructions if task not in completed_tasks]
        task_instructions = task_instructions[len(completed_tasks):]
        utils.delete_error(agent_save_file)
    llm = get_llm_backend(llm_name).run
    agent_cls = get_agent(agent_name)
    agents = [agent_cls(ques, ans, llm, choices, cap, ocr, max_context_len) for ques, choices, ans, cap, ocr in task_instructions]
    # with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    #     executor.map(process_agent_run_step, agents)
    for agent in agents:
        process_agent_run_step(agent)
    # for agent in agents:
        utils.log_agent(agent, agent_save_file)
    print(f'Finished Trial. Total: {len(agents)}')
    
def main():
    for level in ["1-4","5-8","9-12"]:
        run_one_complex_level(level)
    
if __name__ == '__main__':
    main()