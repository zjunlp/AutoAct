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

ZEROSHOT_INSTRUCTION_HOTPOTQA = """I want you to be a good multi-hop question answerer ,solving a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be five types : 
{tools}
(4) Finish[answer], which returns a definite answer. For example, Finish[Richard Nixon] (If it is a judgement question, please Finish[yes] or Finish[no])
(5) Reflect[right/wrong], which reflects the answer right or wrong based on the context history. For example, Reflect[right]
Note that Reflect must be the next Action after Finish. You may take as many steps as necessary.
Question: {question}\n\n{scratchpad}"""

zeroshot_agent_prompt_hotpotqa = PromptTemplate(
                        input_variables=["tools","question", "scratchpad"],
                        template = ZEROSHOT_INSTRUCTION_HOTPOTQA,
                        )

ZEROSHOT_INSTRUCTION_SCIENCEQA = """I want you to be a good multimodal multiple-choice science questions answerer. Select a correct option to a multi-choice multi-modal question with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be five types: 
{tools}
(4) Finish[option], which returns the answer option and finishes the task. For example, Finish[A]
(5) Reflect[right/wrong], which reflects the answer right or wrong based on the context history. For example, Reflect[right]
Note that to determine the answer, it's needed to consider both the Question and the available Options.
Note that Reflect must be the next Action after Finish.
BingSearch and Retrieve can be used multi-times.
Question: {question}\n{scratchpad}"""

zeroshot_agent_prompt_scienceqa = PromptTemplate(
    input_variables=["tools","question", "scratchpad"],
    template = ZEROSHOT_INSTRUCTION_SCIENCEQA,
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


