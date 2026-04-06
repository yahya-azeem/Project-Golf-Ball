import requests
import os
import sys

API_KEY = os.environ.get("RUNPOD_API_KEY")

def check_stock():
    url = "https://rest.runpod.io/v1/gpu-types"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    gpus = r.json()
    
    h100s = [g for g in gpus if "H100" in g['name']]
    for g in h100s:
        print(f"Name: {g['name']}, ID: {g['id']}, Secure Stock: {g.get('secureStock', 'N/A')}, Community Stock: {g.get('communityStock', 'N/A')}")

if __name__ == "__main__":
    check_stock()
