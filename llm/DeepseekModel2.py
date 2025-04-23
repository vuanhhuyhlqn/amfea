from openai import OpenAI
import re
init_idea_text = open("llm/prompts2/init_idea.txt", "r").read()
gen_function_text = open("llm/prompts2/gen_function.txt", "r").read()
reflect_text = open("llm/prompts2/reflect.txt", "r").read()
mutation_text = open("llm/prompts2/mutation.txt", "r").read()

def split_sentences(text):
    lines = text.strip().split('\n')
    sentences = [re.sub(r'^\d+\.\s*', '', line).strip() for line in lines]
    return sentences
def remove_first_last_lines(text):
    lines = text.strip().split('\n')
    result = lines[1:-1]
    return '\n'.join(result)
class DeepsekModel2:
    def __init__(self, API_KEY):
        self.API_KEY = API_KEY

    def init_idea_from_llm(self):
        client = OpenAI(api_key=self.API_KEY, base_url="https://api.deepseek.com")

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": init_idea_text},
            ],
            stream=False
        )
        return split_sentences(response.choices[0].message.content)

    def gen_code_from_idea(self, ideas):
        ideas_text = ""
        for idea in ideas:
            ideas_text += idea + "\n"
        
        u = gen_function_text.format(ideas)

        client = OpenAI(api_key=self.API_KEY, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": u},
            ],
            stream=False
        )
        return remove_first_last_lines(response.choices[0].message.content)
    
    def reflect(self, ideas1, ideas2, per1, per2):
        ideas1_text = ""
        ideas2_text = ""
        for idea in ideas1:
            ideas1_text += idea + "\n"
        for idea in ideas1:
            ideas1_text += idea + "\n"
        prompt = reflect_text.format(ideas1_text, ideas2_text, per1, per2)
        client = OpenAI(api_key=self.API_KEY, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        return split_sentences(response.choices[0].message.content)
    
    def mutation(self, ideas, per):
        ideas_text = ""
        for idea in ideas:
            ideas_text += idea + "\n"
        prompt = mutation_text.format(ideas_text, per)
        client = OpenAI(api_key=self.API_KEY, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        return split_sentences(response.choices[0].message.content)