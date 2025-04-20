import requests

url = "http://localhost:8080/generate"
payload = {"inputs": "rho=1.0 phi=0.0", "parameters": {"max_new_tokens": 32}}
resp = requests.post(url, json=payload, timeout=30)
print(resp.json())

