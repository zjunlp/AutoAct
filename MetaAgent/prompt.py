ACTION_SYSTEM_PROMPT = """In order to complete a complex benchmark, we usually need the collaborative work of the following four types of agents:
1. Plan Agent. This agent is used to plan the specific execution process of the benchmark, solving a given task by determining the order in which other expert language models are invoked;
2. Tool Agent. This type of agents is used to determine how to execute a specific action when solving a task. Tools can include interactive actions with the benchmark environment and actions to call external tools or models outside of the benchmark;
3. Answer Agent. This agent is used to generate the final answer for a given task based on historical information;
4. Reflection Agent. This agent reflects on historical information and answers to determine whether the answer matches the given query.
Above all, the Tool Agent includes many sub-agents that can be flexibly selected. Now your task is to generate 3-5 Tool Agents for solving a given benchmark. Note that all agents are based on language models, and their inputs and outputs must be text. You only need to provide the names and descriptions of the tool agents in order, without any addtional output."""

TOOL_POOL = [
    {"name": "BingSearch","definition":"BingSearch engine can search for rich external knowledge on the Internet based on keywords, which can compensate for knowledge fallacy and knowledge outdated.","usage":"BingSearch[query], which searches the exact detailed query on the Internet and returns the relevant information to the query. Be specific and precise with your query to increase the chances of getting relevant results. For example, Bingsearch[popular dog breeds in the United States]"},
    {"name": "Retrieve", "definition": "Retrieve additional background knowledge crucial for tackling complex problems. It is especially beneficial for specialized domains like science and mathematics, providing context for the task","usage":"Retrieve[entity], which retrieves the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to retrieve. For example, Retrieve[Milhouse]"},
    {"name": "Lookup", "definition": "A Lookup Tool returns the next sentence containing the target string in the page from the search tool (like BingSearch or Retrieve),so it is recommended to use with Bingsearch and Retrieve, simulating Ctrl+F functionality on the browser to find target answer.","usage":"Lookup[keyword], which returns the next sentence containing the keyword in the last passage successfully found by Retrieve or BingSearch. For example, Lookup[river]."},
    {"name": "Image2Text", "definition":"Image2Text is used to detect words in images convert them into text by OCR and generate captions for images. It is particularly valuable when understanding an image semantically, like identifying objects and interactions in a scene.","usage":"Image2Text[image], which generates captions for the image and detects words in the image. You are recommended to use it first to get more information about the image to the question. If the question contains an image, it will return the caption and OCR text, else, it will return None. For example, Image2Text[image]."},
    {"name": "Text2Image","definition":"Text2Image Specializes in converting textual information into visual representations, facilitating the incorporation of textual data into image-based formats within the task.","usage":"Text2Image[text], which generates an image for the text provided by using multimodal models. For example, Text2Image[blue sky]"},
    {"name": "KnowledgeGraph","definition":"KnowledgeGraph is used to query knowledge graph and get the query results as output.","usage":"KnowledgeGraph[query], which queries the knowledge graph and returns the relevant information to the query. For example, KnowledgeGraph[What is the capital of China?]"},
    {"name": "Database", "definition": "A Database tool can output any valid SQL commmands to finish databse query, update task.","usage":"Database[query], which outputs any valid SQL commmands to finish databse query, update task. For example, Database[SELECT * FROM Customers WHERE Country='Mexico';]"},
    {"name": "Calculator", "definition": "Calculator is used to calculate the result of the given mathematical expression.","usage":"Calculator[query], which calculates the result of the given mathematical expression. For example, Calculator[2+2]"},
    {"name": "Table Verbalizer", "definition": "Table Verbalizer is used to convert structured tables into text to enhance the comprehension of tabular information.","usage":"Table Verbalizer[table], which converts structured tables into text to enhance the comprehension of tabular information. For example, Table Verbalizer[table]"},
    {"name": "Code Interpreter", "definition": "Code Interpreter is a tool or software that interprets and executes code written in Python. It analyzes the source code line by line and translates it into machine-readable instructions or directly executes the code and returns Execution results","usage":"Code[python], which interprets and executes Python code, providing a line-by-line analysis of the source code and translating it into machine-readable instructions. For instance, Code[print(\"hello world!\")]"}
    ]

TASK_PROMPT_TEMPLATE = """The following is the given task name and description , and you need to select three most corresponding tool agents according to the above rules in the format of one line one tool.
Here are tools to be selected from: {tool_pool}
Task Name: {task_name}
Task Description: {task_description}
Task Tool Agents: 
"""


SCIENCEQA_DESCRIPTION = """This is a multimodal question answering benchmark that requires a language model to use tools to convert image information into textual information. At the same time, this benchmark contains rich background knowledge, so language model needs to obtain external knowledge to enhance the understanding of the benchmark."""

HOTPOTQA_DESCRIPTION = """This is a question answering benchmark"""



"""
In order to complete a complex benchmark, we usually need the collaborative work of the following four types of agents:
1. Plan Agent. This agent is used to plan the specific execution process of the benchmark, solving a given task by determining the order in which other expert language models are invoked;
2. Action Agent. This type of agents is used to determine how to execute a specific action when solving a task. Actions can include interactive actions with the benchmark environment and actions to call external tools or models outside of the benchmark;
3. Answer Agent. This agent is used to generate the final answer for a given task based on historical information;
4. Reflection Agent. This agent reflects on historical information and answers to determine whether the answer matches the given query.

Above all, the Action Agent includes the following sub-agents that can be selected:
{"name": "Look up", "definition": "A Look up agent return the next sentence contain the target string in the page from search agent, simulating Ctrl+F functionality on the browser"},
{"name": "Click", "definiton": "A Click agent would choose from all clickable buttons or urls to requeset a new web page."},
{"name": "Database", "definition": "A Database agent can output any valid SQL commmands to finish databse query, update task."},
{"name": "Bash", "definition": "Bash agent would generate bash commands and collection the output from systems."},
{"name": "Knowledge Graph", "definition": "A Knowledge Graph agent uses basic KG-querying tools and get the query results as output."},
{"name": "Knowledge Retrieval", "definition": "Retrieves additional background knowledge crucial for tackling complex problems. It is especially beneficial for specialized domains like science and mathematics, providing context for the task."},
{"name": "Image Captioner", "definition": "Designed to generate captions for images. It is particularly valuable when understanding an image semantically, like identifying objects and interactions in a scene."}, {"name": "Text Detector", "definition": "Text detector is used to detect words in images and convert it into text."},
{"name": "Search Engine", "definition": "Search engine can search for rich knowledge on the internet based on keywords, which can compensate for knowledge fallacy and knowledge outdated."},
{"name": "Program Generator", "definition": "Program Generator generates Python programs to solve queries effectively, which is particularly beneficial for queries requiring complex computations or intricate logical operations."},
{"name": "Table Verbalizer", "definition": "Table Verbalizer is used to convert structured tables into text to enhance the comprehension of tabular information."},

Now your task is to select proper Action Agents for solving a given benchmark. Note that you only need to provide the names of the action agents in order, without any addtional output.
"""

"""The following is the given benchmark name and description, and you need to select almost 3 corresponding action agents in the format of one line one agent. Benchmark Name: WebShop Benchmark Description: WebShop is a recently proposed online shopping website environment with 1.18M real-world products and human instructions. Benchmark Action Agents:"""


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

BENCHMARK_DESCRIPTION = {
    "ScienceQA": SCIENCEQA_TASK_DESCRIPTION,
    "HotpotQA": HOTPOTQA_TASK_DESCRIPTION,
}