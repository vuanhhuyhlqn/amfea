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

reverse_file = open("llm/prompts/reverse.txt", "r")
reverse_text = reverse_file.read()

def split_prompts_deekseek(response):
    descriptions = [line.strip() for line in response.strip().split('\n') if line.strip()]
    ideas = []
    for desc in descriptions:
        ideas.append(desc)
    return ideas

def extract_code(response):
    if response.startswith("<") and response.endswith(">"):
        response = response[1:-1].strip()
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

        # self.messages.append({"role": "user", "content": init_prompt})
        # response = self.client.chat.completions.create(
        #     model=self.model,
        #     messages=self.messages,
        #     temperature=self.temperature,
        # )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": init_prompt},
            ],
            max_tokens=1024,
            temperature=0.7,
            stream=False
        )

        ideas = split_prompts_deekseek(response.choices[0].message.content)


        return ideas
    
    def idea_to_code_function(self, idea):
        print("Creating code...")
        create_prompt = create_text.format(idea)

        # self.messages.append({"role": "user", "content": create_prompt})
        # response = self.client.chat.completions.create(
        #     model=self.model,
        #     messages=self.messages,
        #     temperature=self.temperature,
        # )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": create_prompt},
            ],
            max_tokens=1024,
            temperature=0.7,
            stream=False
        )

        # print("Response: ", response.choices[0].message.content)

        code = extract_code(response.choices[0].message.content)
        return code
    
    def crossover(self, p1_idea, p2_idea, p1_performance, p2_performance):
        print("Crossover...")
        crossover_prompt = crossover_text.format(p1_idea, p1_performance ,p2_idea, p2_performance)

        # self.messages.append({"role": "user", "content": crossover_prompt})
        # response = self.client.chat.completions.create(
        #     model=self.model,
        #     messages=self.messages,
        #     temperature=self.temperature,
        # )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": crossover_prompt},
            ],
            max_tokens=1024,
            temperature=0.7,
            stream=False
        )

        return response.choices[0].message.content.strip()
    
    def mutation(self, idea, performance):
        print("Mutation...")
        mutation_prompt = mutation_text.format(idea, performance)

        # self.messages.append({"role": "user", "content": mutation_prompt})
        # response = self.client.chat.completions.create(
        #     model=self.model,
        #     messages=self.messages,
        #     temperature=self.temperature,
        # )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": mutation_prompt},
            ],
            max_tokens=1024,
            temperature=0.7,
            stream=False
        )

        return response.choices[0].message.content.strip()
    
    def reverse(self, idea):
        print("Reverse...")
        reverse_prompt = reverse_text.format(idea)

        # self.messages.append({"role": "user", "content": reverse_prompt})
        # response = self.client.chat.completions.create(
        #     model=self.model,
        #     messages=self.messages,
        #     temperature=self.temperature,
        # )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": reverse_prompt},
            ],
            max_tokens=1024,
            temperature=0.7,
            stream=False
        )

        return response.choices[0].message.content.strip()