"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: Apache License 2.0
 For full license text, see the LICENSE file in the repo root or https://www.apache.org/licenses/LICENSE-2.0
"""

from langchain.prompts import PromptTemplate

# ZEROSHOT_INSTRUCTION = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
# (1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search. For example, Search[Milhouse]
# (2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search. For example, Lookup[named after]
# (3) Finish[answer], which returns the answer and finishes the task. For example, Finish[Richard Nixon] 
# You may take as many steps as necessary.
# Question: {question}{scratchpad}"""

ZEROSHOT_INSTRUCTION = """I want you to be a good multimodal multiple-choice science questions answerer. Select a correct option to a multi-choice multi-modal question with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be five types: 
(1) Image2Text[image], which generates captions for the image and detects words in the image.You are recommand to use it first to get more information about the imgage to the question. If the questions contains image, it will return catption and ocr text, else, it will return None. For example, ImageCaptioner[image]
(2) BingSearch[question], which searches the exact detailed question on the Internet and returns the relevant information to the query. Be specific and precise with your query to increase the chances of getting relevant results. For example, instead of searching for "dogs," you can search for "popular dog breeds in the United States." For example, BingSearch[Which type of computer networking technology, developed in the 1970s, allows devices to communicate over a shared network]
(3) Retrieve[entity], which retrieves the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to retrieve. For example, Retrieve[Milhouse]
(4) Finish[option], which returns the answer option and finishes the task. For example, Finish[A]
(5) Reflect[right/wrong], which reflects the answer right or wrong based on the context history. For example, Reflect[right]
Note that to determine the answer, it's needed to consider both the Question and the available Options.
Note that Reflect must be the next Action after Finish.
BingSearch and Retrieve can be used multi-times.
Question: {question}\n{scratchpad}"""

zeroshot_agent_prompt = PromptTemplate(
    input_variables=["question", "scratchpad"],
    template = ZEROSHOT_INSTRUCTION,
)

REACT_INSTRUCTION = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
You may take as many steps as necessary.
Here are some examples:
{examples}
(END OF EXAMPLES)
Question: {question}{scratchpad}"""

react_agent_prompt = PromptTemplate(
                        input_variables=["examples", "question", "scratchpad"],
                        template = REACT_INSTRUCTION,
                        )

PLAN_INSTRUCTION = """Setup a plan for answering question with Actions. Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
{examples}
(END OF EXAMPLES)
Question: {question}
Plan:"""

plan_prompt = PromptTemplate(
                input_variables=["examples", "question"],
                template = PLAN_INSTRUCTION,
            )

PLANNER_INSTRUCTION = """Solve a question answering task with Plan, interleaving Action, Observation steps. Plan is decided ahead of Actions. Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
You may take as many steps as necessary.
Here are some examples:
{examples}
(END OF EXAMPLES)
Question: {question}
Plan: {plan}{scratchpad}"""

planner_agent_prompt = PromptTemplate(
                        input_variables=["examples", "question", "plan", "scratchpad"],
                        template = PLANNER_INSTRUCTION,
                        )

PLANNERREACT_INSTRUCTION = """Solve a question answering task with Plan, interleaving Thought, Action, Observation steps. Plan is decided ahead of Actions. Thought can reason about the current situation. Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.
(3) Finish[answer], which returns the answer and finishes the task.
You may take as many steps as necessary.
Here are some examples:
{examples}
(END OF EXAMPLES)
Question: {question}
Plan: {plan}{scratchpad}"""

plannerreact_agent_prompt = PromptTemplate(
                        input_variables=["examples", "question", "plan", "scratchpad"],
                        template = PLANNERREACT_INSTRUCTION,
                        )


