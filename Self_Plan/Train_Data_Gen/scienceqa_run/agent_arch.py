"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: Apache License 2.0
 For full license text, see the LICENSE file in the repo root or https://www.apache.org/licenses/LICENSE-2.0
"""

import random
import re, string, os
import json 
import time
import tiktoken
import numpy as np
from langchain.llms.base import BaseLLM
from langchain import OpenAI, Wikipedia
from langchain.docstore.base import Docstore
from langchain.agents.react.base import DocstoreExplorer
from langchain.prompts import PromptTemplate
from collections import Counter

from scienceqa_run.pre_prompt import (react_agent_prompt, zeroshot_agent_prompt, 
                         plan_prompt, planner_agent_prompt, plannerreact_agent_prompt)
from scienceqa_run.fewshots import REACT_EXAMPLE, PLANNER_EXAMPLE, PLAN_EXAMPLE, PLANNERREACT_EXAMPLE

from web_run.llms import token_enc
from scienceqa_run.utils import call_bing_search, parse_bing_result

def parse_action(string):
    pattern = r'^(\w+)\[(.+)\]$'
    match = re.match(pattern, string)
    
    if match:
        action_type = match.group(1)
        argument = match.group(2)
        return action_type, argument
    else:
        action_type, argument = fuzzy_parse_action(string)
        return action_type, argument
        
def fuzzy_parse_action(text):
    text = text.strip(' ').strip('.')
    pattern = r'^(\w+)\[(.+)\]'
    match = re.match(pattern, text)
    if match:
        action_type = match.group(1)
        argument = match.group(2)
        return action_type, argument
    else:
        return text, ''

def format_step(step: str) -> str:
    step = step.strip('\n').strip().replace('\n', '')
    if step.startswith("Thought") or step.startswith("Action"):
        step = step.split()[2:]
        step = " ".join(step)
    if "Thought" in step:
        step = step.split("Thought")[0].strip()
    if "Action" in step:
        step = step.split("Action")[0].strip()
    if "Observation" in step:
        step = step.split("Observation")[0].strip()
    return step

def truncate_scratchpad(scratchpad: str, n_tokens: int = 1600, tokenizer = token_enc) -> str:
    lines = scratchpad.split('\n')
    observations = filter(lambda x: x.startswith('Observation'), lines)
    observations_by_tokens = sorted(observations, key=lambda x: len(tokenizer.encode(x)))
    while len(token_enc.encode('\n'.join(lines))) > n_tokens:
        largest_observation = observations_by_tokens.pop(-1)
        ind = lines.index(largest_observation)
        lines[ind] = largest_observation.split(':')[0] + ': [truncated wikipedia excerpt]'
    return '\n'.join(lines)

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    
    def white_space_fix(text):
        return " ".join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
  
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def EM(answer, key) -> bool:
    return normalize_answer(answer) == normalize_answer(key)

def score_string_similarity(str1, str2):
    if str1 == str2:
        return 2.0
    elif " " in str1 or " " in str2:
        str1_split = str1.split(" ")
        str2_split = str2.split(" ")
        overlap = list(set(str1_split) & set(str2_split))
        return len(overlap) / max(len(str1_split), len(str2_split))
    else:
        return 0.0
    
def normalize_ground_scienceqa(gt_ans):
    gt_ans = gt_ans.lower()
    return gt_ans
    
def normalize_prediction_scienceqa(prediction, options=None):
    # the string answer from choices
    if options:
        options = [x.lower() for x in options]
        if prediction is None:
            prediction = options[0]
        elif isinstance(prediction, str):
            if prediction not in options:
                # find the most similar option
                scores = [score_string_similarity(x, prediction) for x in options]
                max_idx = int(np.argmax(scores)) # json does not recognize NumPy data types
                prediction = options[max_idx]
    return prediction


class BaseAgent:
    def __init__(self,
                 question: str,
                 key: str,
                 llm: BaseLLM,
                 context_len: int = 2000,
                 max_steps: int= 15,
                 docstore: Docstore = Wikipedia()
                 ) -> None:
        
        self.question = question
        self.answer = ''
        self.key = key
        self.max_steps = max_steps
        self.agent_prompt = ""
        self.examples = ""
        self.context_len = context_len
        self.run_error = False
        self.name = "Base_HotPotQA_run_Agent"

        self.docstore = DocstoreExplorer(docstore) # Search, Lookup
        # self.bing = BingSearch()
        self.llm = llm
        # self.action_agents = {
        #     "retrieve": agent;
        # }
        
        self.enc = token_enc
        self.__reset_agent()
    
    def run(self, reset = True) -> None:
        if reset:
            self.__reset_agent()
        
        print(self.question)
        while not self.is_halted() and not self.is_finished() and not self.run_error:
            self.step()
    
    def prompt_agent(self) -> str:
        generation = self.llm(self._build_agent_prompt())
        self.check_run_error(generation)
        return format_step(generation)
 
    def check_run_error(self, text):
        if text in ["No response"]:
            self.run_error = True
            
    def is_finished(self) -> bool:
        return self.finished
    
    def reward(self) -> float:
        return f1_score(self.answer, self.key)   
    
    def is_correct(self) -> bool:
        return EM(self.answer, self.key)

    def is_halted(self) -> bool:
        return ((self.step_n > self.max_steps)
                or (len(self.enc.encode(self._build_agent_prompt())) > self.context_len)
                ) and not self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.finished = False
        self.scratchpad: str = ''

    def set_qa(self, question: str, key: str) -> None:
        self.question = question
        self.key = key

    def _think(self):
        self.scratchpad += f'\nThought {self.step_n}:'
        thought = self.prompt_agent()
        self.scratchpad += ' ' + thought
        print(self.scratchpad.split('\n')[-1])
    
    def _action(self):
        self.scratchpad += f'\nAction {self.step_n}:'
        action = self.prompt_agent()
        self.scratchpad += ' ' + action
        action_type, argument = parse_action(action)
        print(self.scratchpad.split('\n')[-1])
        return action_type, argument
        
    def step(self) -> None:
        
        # agent forward
        ret = self.forward()
        if ret:
            action_type, argument = ret[0], ret[1]
        else:
            action_type = ret
        
        # Observe
        self.scratchpad += f'\nObservation {self.step_n}: '
        
        if action_type == 'Finish':
            self.answer = argument
            if self.is_correct():
                self.scratchpad += 'Answer is CORRECT'
            else: 
                self.scratchpad += 'Answer is INCORRECT'
            self.finished = True
            self.step_n += 1
            return

        if action_type == 'Retrieve':
            try:
                self.scratchpad += format_step(self.docstore.search(argument))
            except Exception as e:
                print(e)
                self.scratchpad += f'Could not find that page, please try again.'
            
        elif action_type == 'Lookup':
            try:
                self.scratchpad += format_step(self.docstore.lookup(argument))
            except ValueError:
                self.scratchpad += f'The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given.'
        elif action_type == 'BingSearch':
            try:
                responses = call_bing_search("https://api.bing.microsoft.com/v7.0/search","5fc96b4311e74e519dae3c2acfa56ef8",argument,count=1)
                responses = parse_bing_result(responses)
                if len(responses)>0 :
                    self.scratchpad += format_step(responses[0])
                else:
                    self.scratchpad += "Bing search error"
            except :
                self.scratchpad+= f'Search error,please try again'
        else:
            self.scratchpad += 'Invalid Action. Valid Actions are Lookup[<topic>] Retrieve[<topic>] BingSearch[<topic>] and Finish[<answer>].'
            # self.scratchpad += 'Invalid Action. You can only generate Lookup[<topic>], Search[<topic>] or Finish[<answer>].'

        print(self.scratchpad.split('\n')[-1])

        self.step_n += 1
    
    def _build_agent_prompt(self) -> str:
        raise NotImplementedError
    
    def forward(self):
        raise NotImplementedError
class ZeroshotThinkAgent(BaseAgent):
    def __init__(self,
                 question: str,
                 key: str,
                 llm,
                 choices,
                 caption,
                 ocr,
                 context_len: int = 4096
                 ) -> None:
        super().__init__(question, key, llm, context_len)

        self.examples = ""
        self.agent_prompt = zeroshot_agent_prompt
        self.name = "ZeroshotThink_HotPotQA_run_Agent"
        self.last_step = ""
        self.bingresults = []
        self.bingindex = 0
        self.bingkeyword = ""
        self.inds = ["A", "B", "C", "D", "E"]
        self.choices = choices
        self.key = self.choices[self.inds.index(key)]
        self.caption = caption
        self.ocr = ocr

    def reward(self) -> float:
        if self.answer in self.inds[:len(self.choices)]:
            prediction = self.choices[self.inds.index(self.answer)]
        else:
            prediction = normalize_prediction_scienceqa(self.answer.lower(), self.choices)

        return prediction.strip().lower() == self.key.strip().lower()
        
    def is_correct(self) -> bool:
        return self.reward()
    
    def forward(self):
        self._think()
        action_type, argument = self._action()
        return action_type, argument

    def step(self) -> None:
        
        # agent forward
        ret = self.forward()
        if ret:
            action_type, argument = ret[0], ret[1]
        else:
            action_type = ret
        
        # Observe
        self.scratchpad += f'\nObservation {self.step_n}: '
        
        if action_type == 'Finish':
            if self.answer != '':
                self.answer = argument
                if self.is_correct():
                    self.scratchpad += 'Answer is CORRECT'
                else: 
                    self.scratchpad += 'Answer is INCORRECT'
                self.finished = True
                self.step_n += 1
                return
            else:
                self.answer = argument
                self.scratchpad += 'Please reflect your answer.'

        elif action_type == 'Reflect':
            if self.last_step != "Finish":
                self.scratchpad += f'Invalid action. Reflect only follows the Finish action.'
            else:
                if argument.lower() == 'right':
                    if self.is_correct():
                        self.scratchpad += 'Answer is CORRECT'
                    else: 
                        self.scratchpad += 'Answer is INCORRECT'
                    self.finished = True
                    self.step_n += 1
                    print(self.scratchpad.split('\n')[-1])
                    return
                else:
                    self.scratchpad += f'The answer maybe wrong, please try to solve the question again.'
        elif action_type == 'Retrieve':
            try:
                self.scratchpad += format_step(self.docstore.search(argument))
            except Exception as e:
                print(e)
                self.scratchpad += f'Could not find that page, please try again.'
        #add Bing_search
        elif action_type == 'BingSearch':
            try:
                responses = call_bing_search(query=argument, count=4)
                self.bingresults = responses
                if len(responses) > 0 :
                    for r in responses:
                        self.scratchpad += format_step(r)
                else:
                    self.scratchpad += "BingSearch error, please try again."
            except :
                self.scratchpad+= f'BingSearch error, please try again.'
        elif action_type == 'Image2Text':
            caption = self.caption
            ocr = self.ocr
            if caption == "no img" or ocr == "no img":
                self.scratchpad += "This question doesn't have an image."
            else:
                self.scratchpad += format_step(f"Image caption: {caption} OCR result: {ocr}")
        else:
            self.scratchpad += 'Invalid Action. Valid Actions are Retrieve[<topic>] BingSearch[<topic>] Finish[<answer>] Image2Text[imgae] and Reflect[right/wrong].'
            # self.scratchpad += 'Invalid Action. You can only generate Lookup[<topic>], Search[<topic>] or Finish[<answer>].'
        self.last_step = action_type
        print(self.scratchpad.split('\n')[-1])

        self.step_n += 1

    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
                            question = self.question,
                            scratchpad = self.scratchpad,
                            )
        

def get_agent(agent_name):
    if agent_name in ["ZeroshotThink_ScienceQA_run_Agent"]:
        return ZeroshotThinkAgent