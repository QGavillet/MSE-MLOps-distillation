import requests

# Replace 'path/to/your/test_image.jpg' with the actual path to an image file on your system.
with open("test/test_image.jpg", "rb") as f:
    image_bytes = f.read()

# The deployment is assumed to be running at http://localhost:8000/
# Make sure the deployment is running before sending the request.
response = requests.post("http://localhost:8000/", data=image_bytes, headers={"Content-Type": "application/octet-stream"})

# Print out the prediction result from the server
print(response.text)
