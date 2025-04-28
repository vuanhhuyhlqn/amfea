from openai import OpenAI
from .AbstractModel import AbstractModel
import json
import time
import threading

rate_limit_lock = threading.Lock()
last_call_time = 0

def rate_limited(calls_per_minute):
    interval = 60.0 / calls_per_minute
    def decorator(func):
        def wrapper(*args, **kwargs):
            global last_call_time
            with rate_limit_lock:
                current_time = time.time()
                wait_time = last_call_time + interval - current_time
                if wait_time > 0:
                    time.sleep(wait_time)
                result = func(*args, **kwargs)
                last_call_time = time.time()
                return result
        return wrapper
    return decorator

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

    @rate_limited(4)
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
    
    @rate_limited(4)
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

    @rate_limited(4)
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
    
    @rate_limited(4)
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
    
    @rate_limited(4)
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