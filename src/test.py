# File name: model_client.py
import requests

english_text = "Hello world! How are you doing today?"

response = requests.post("http://localhost:8000/", json=english_text)
french_text = response.text

print(french_text)