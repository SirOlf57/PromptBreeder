import json

import requests
from colorama import Fore


class OllamaClient:
    def __init__(self, model: str = "llama3:8b"):
        self.model = model

    def prompt(self, prompt, temperature=0.0):
        print(Fore.GREEN, prompt)

        url = "http://localhost:11434/api/generate"
        headers = {"Content-Type": "application/json"}

        data = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": 1024,
            "stream": False
        }
        response = {}
        try:
            response = requests.post(url=url, headers=headers, data=json.dumps(data))
            print(Fore.WHITE, json.loads(response.text)["response"])

            return json.loads(response.text)["response"]

        except Exception as e:
            print("Error: ", response.status_code, response.text)

        return response