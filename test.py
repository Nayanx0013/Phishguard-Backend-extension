import requests

response = requests.post(
    "http://localhost:5000/predict",
    json={"url": "http://paypal.secure-login.tk/verify"}
)
print(response.json())