# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#
# TO MAKE THIS TEST WORK, YOU NEED TO
#   - make sure the server is running
#   - forward the port 8000 to have access locally
#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

import requests

english_text = "Hello world! How are you doing today?"

response = requests.post("http://localhost:8000/", json=english_text)
french_text = response.text

print(french_text)