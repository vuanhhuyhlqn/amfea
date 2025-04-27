from openai import OpenAI
from .AbstractModel import AbstractModel
import json

init_file = open("llm/matrix_prompts/initial.txt", "r")
init_text = init_file.read()

create_file = open("llm/matrix_prompts/create.txt", "r")
create_text = create_file.read()

crossover_file = open("llm/matrix_prompts/crossover.txt", "r")
crossover_text = crossover_file.read()

mutation_file = open("llm/matrix_prompts/mutation.txt", "r")
mutation_text = mutation_file.read()

reverse_file = open("llm/matrix_prompts/reverse.txt", "r")
reverse_text = reverse_file.read()

def split_prompts(response_content):
    try:
        data = json.loads(response_content)
        strategies = data.get("strategies", [])
        if not isinstance(strategies, list):
            raise ValueError("Invalid format: 'strategies' should be a list.")
        return strategies
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return []
    
def clean_code_output(response_content):
    cleaned = response_content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split('\n', 1)[-1] 
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit('\n', 1)[0] 
    return cleaned.strip()

class GPTModel(AbstractModel):
    def __init__(self, api_key, model, temperature):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.client = OpenAI(api_key=self.api_key)

    def initial_strategies(self, num_strategies):
        print("Creating {0} strategies".format(num_strategies))

        init_prompt = init_text.format(num_strategies)
        # print(init_prompt)

        response = self.client.chat.completions.create(
            model = self.model,
            messages = [
                {"role": "user", "content": init_prompt},
            ],
            max_tokens = 1024,
            temperature = self.temperature,
            stream = False
        )
        # print(response.choices[0].message.content)

        strategies = split_prompts(response.choices[0].message.content)
        return strategies
    
    def strategy_to_code(self, strategy):
        print("Creating code...")
        create_prompt = create_text.format(strategy)

        response = self.client.chat.completions.create(
            model = self.model,
            messages = [
                {"role": "user", "content": create_prompt},
            ],
            max_tokens = 1024,
            temperature = self.temperature,
            stream = False
        )

        # print(response.choices[0].message.content)
        code = clean_code_output(response.choices[0].message.content)
        return code

    def crossover(self, p1_strategy, p2_strategy, p1_performance, p2_performance):
        print("Crossover...")
        crossover_prompt = crossover_text.format(p1_strategy, p1_performance ,p2_strategy, p2_performance)
        # print(crossover_prompt)

        response = self.client.chat.completions.create(
            model = self.model,
            messages = [
                {"role": "user", "content": crossover_prompt},
            ],
            max_tokens = 1024,
            temperature = self.temperature,
            stream = False
        )
        
        # print(response.choices[0].message.content)
        return response.choices[0].message.content.strip()
    
    def mutation(self, strategy, performance):
        print("Mutation...")
        mutation_prompt = mutation_text.format(strategy, performance)
        # print(mutation_prompt)

        response = self.client.chat.completions.create(
            model = self.model,
            messages = [
                {"role": "user", "content": mutation_prompt},
            ],
            max_tokens = 1024,
            temperature = self.temperature,
            stream = False
        )

        return response.choices[0].message.content.strip()
    
    def reverse(self, strategy):
        print("Reverse...")
        reverse_prompt = reverse_text.format(strategy)
        # print(reverse_prompt)

        response = self.client.chat.completions.create(
            model = self.model,
            messages = [
                {"role": "user", "content": reverse_prompt},
            ],
            max_tokens = 1024,
            temperature = self.temperature,
            stream = False
        )

        return response.choices[0].message.content.strip()