import os
import copy
from langchain import OpenAI, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    AIMessage
)


class MetaAgent:

    def __init__(
        self,
        model_name: str,
        openai_key: str = "EMPTY",
        url: str = "http://localhost:8000/v1",
        system_prompt: str = None,
    ):
        self.key = openai_key
        self.url = url
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.llm = self._get_llm()
        self.prompt = self._init_prompt()
        self.prompt_args = dict()
        
    def _get_llm(self):
        os.environ['OPENAI_API_KEY'] = self.key
        if "text" in self.model_name:
            llm = OpenAI(model=self.model_name)
        elif "gpt" in self.model_name:
            llm = ChatOpenAI(model=self.model_name)
        else:
            os.environ['OPENAI_API_BASE'] = self.url
            llm = ChatOpenAI(model=self.model_name)
        return llm
    
    def _init_prompt(self):
        if not self.system_prompt:
            return []

        if isinstance(self.llm, ChatOpenAI):
            system_prompt_template = SystemMessagePromptTemplate.from_template(self.system_prompt)
            return [system_prompt_template]
        else:
            system_prompt_template = "system: " + self.system_prompt
            return [system_prompt_template]
        
    def generate(
        self,
        human_prompt_template,
        human_prompt_args,
        temprature=0.2,
        top_k=40,
        top_p=0.75,
        max_tokens=512,
        stop=None,
        update_prompt=False,
        reset_prompt=False
    ):
        _old_prompt = copy.deepcopy(self.prompt)
        _old_prompt_args = copy.deepcopy(self.prompt_args)
        self.prompt_args.update(human_prompt_args)
        
        if isinstance(self.llm, ChatOpenAI):
            human_prompt_template = HumanMessagePromptTemplate.from_template(human_prompt_template)
            self.prompt.append(human_prompt_template)
            prompt_template = ChatPromptTemplate.from_messages(self.prompt)
            prompt = prompt_template.format_messages(**self.prompt_args)
        else:
            self.prompt.append("human: " + human_prompt_template)
            prompt_template = PromptTemplate.from_template("\n\n".join(self.prompt))
            prompt = prompt_template.format_prompt(**self.prompt_args)
        
        response = self.llm(
            prompt,
            temprature=temprature,
            top_k=top_k,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop
        )
        
        output = response.content if isinstance(response, AIMessage) else response
        
        if update_prompt:
            if isinstance(self.llm, ChatOpenAI):
                ai_prompt_template = AIMessagePromptTemplate.from_template(output)
                self.prompt.append(ai_prompt_template)
            else:
                self.prompt.append(output)
        else:
            self.prompt = _old_prompt
            self.prompt_args = _old_prompt_args
            
        if reset_prompt:
            self._init_prompt()

        return output