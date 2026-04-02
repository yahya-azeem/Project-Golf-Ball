import requests
import os
import sys
import json

def add_registry(api_key, name, username, password):
    """
    Adds a container registry to RunPod via the REST API.
    Returns the ID of the new registry.
    """
    url = "https://rest.runpod.io/v1/containerregistryauth"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "name": name,
        "username": username,
        "password": password
    }

    print(f"Adding registry: {name} (user: {username})...")
    try:
        r = requests.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        reg_id = data.get("id")
        print(f"Successfully added registry. ID: {reg_id}")
        return reg_id
    except Exception as e:
        print(f"Error adding registry: {e}", file=sys.stderr)
        # Attempt to list if it already exists
        print("Checking if registry already exists...")
        try:
            rl = requests.get(url, headers=headers)
            rl.raise_for_status()
            registries = rl.json()
            existing = next((rg for rg in registries if rg['name'] == name), None)
            if existing:
                print(f"Found existing registry ID: {existing['id']}")
                return existing['id']
        except:
             pass
        sys.exit(1)

if __name__ == "__main__":
    runpod_key = os.environ.get("RUNPOD_API_KEY")
    gh_pat = os.environ.get("GH_PAT_SUBMISSION") # The PAT provided by the user
    
    if not runpod_key:
        print("Error: RUNPOD_API_KEY not set.")
        sys.exit(1)
        
    if not gh_pat:
        print("Error: GH_PAT_SUBMISSION not set.")
        sys.exit(1)

    reg_id = add_registry(
        runpod_key, 
        "GitHub-GHCR", 
        "yahya-azeem", 
        gh_pat
    )
    
    # Save ID to a temporary file for the next script to read if needed
    with open("runpod_registry_id.txt", "w") as f:
        f.write(str(reg_id))
