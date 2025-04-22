from openai import OpenAI

class DeepseekModel:
    def __init__(self, api_key, model, temperature):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.alpha = 0.1
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepinfra.com/v1/openai")
        self.messages = []
        
    def increase_temperature(self):
        self.temperature += self.alpha