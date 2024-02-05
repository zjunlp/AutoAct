HOTPOTQA_TASK_NAME = "HotpotQA"
HOTPOTQA_TASK_DESCRIPTION = "This is a question-answering task that includes high-quality multi-hop questions and do not contain images. It tests language modeling abilities for multi-step reasoning and covers a wide range of topics. Some questions are challenging, while others are easier, requiring multiple steps of reasoning to arrive at the final answer."

SCIENCEQA_TASK_NAME = "ScienceQA"
SCIENCEQA_TASK_DESCRIPTION = " This is a multimodal question-answering task that necessitates a model to utilizetools for transforming image information intotextual data. Simultaneously, this task incorporates substantial background knowledge, requiring the language model to acquire external information to enhance its comprehension of the task"


DATA_GEN_SYSTEM_PROMPT  = """I want you to be a QA pair generator to generate high-quality questions for use in Task
described as follows :
Task Name: {task_name}
Task Description: {task_description}
"""

HOTPOTQA_DATA_GEN_HUMAN_PROMPT = """{QA_pairs}\nModelled on all examples above,I want you to generate new different {Gen_num} Question-Answer pairs. The format like below:
Question: The Treaty of Versailles, signed in 1919, officially ended which war?
Answer: World War I
"""

SCIENCEQA_DATA_GEN_HUMAN_PROMPT = """{QA_pairs}\nModelled on all examples above,I want you to generate {Gen_num} new different  multimodal multiple-choice science questions.The question format like below:
Question: Which of these states is farthest north?
Options: Options: (A) West Virginia (B) Louisiana (C) Arizona (D) Oklahoma   (hint:ensure all choices in one line ,not one choice one line)
Ocr: Oklahoma,West Virginia,Arizona,Louisiana
Caption: An aerial view of a painting of a forest
Answer: A. West Virginia
"""
