import requests

API_KEY = "pa-AfPS41qzQDNg9EXkhUbE9H0F4gs51IgRVdV6LYaLqHP"
MODEL = "voyage-lite-01"

text = "Hello, this is a test embedding."

response = requests.post(
    "https://api.voyageai.com/v1/embeddings",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={
        "model": MODEL,
        "input": text
    },
)

print("STATUS:", response.status_code)
print("RESPONSE:")
print(response.json())
