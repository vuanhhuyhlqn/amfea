from openai import OpenAI
from .AbstractModel import AbstractModel

init_file = open("llm/prompts/initial.txt", "r")
init_text = init_file.read()

create_file = open("llm/prompts/create.txt", "r")
create_text = create_file.read()

crossover_file = open("llm/prompts/crossover.txt", "r")
crossover_text = crossover_file.read()

mutation_file = open("llm/prompts/mutation.txt", "r")
mutation_text = mutation_file.read()

def split_prompts_deekseek(response):
    descriptions = [line.strip() for line in response.strip().split('\n') if line.strip()]
    ideas = []
    for desc in descriptions:
        ideas.append(desc)
    return ideas

def extract_code_from_markdown(response):
    if response.startswith("```python"):
        response = response[len("```python"):].strip()
    if response.endswith("```"):
        response = response[:-len("```")].strip()
    return response

class DeepseekModel(AbstractModel):
    def __init__(self, api_key, model, temperature):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.alpha = 0.1
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
        self.messages = []
        
    def increase_temperature(self):
        self.temperature += self.alpha

    def initial_ideas(self, num_ideas):
        print("Creating {0} ideas".format(num_ideas))

        init_prompt = init_text.format(num_ideas)

        self.messages.append({"role": "user", "content": init_prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
        )

        print("Response: ", response.choices[0].message.content)

        ideas = split_prompts_deekseek(response.choices[0].message.content)

        return ideas
    
    def idea_to_code_function(self, idea):
        create_prompt = create_text.format(idea)

        self.messages.append({"role": "user", "content": create_prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
        )

        code = response.choices[0].message.content.strip()
        return extract_code_from_markdown(code)