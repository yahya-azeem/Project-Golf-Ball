import requests
import os
import sys
import json

def upsert_template(api_key, name, image_name):
    """
    Creates or updates a RunPod template using the REST API.
    """
    base_url = "https://api.runpod.io/v1/templates"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 1. Fetch existing templates
    try:
        r_list = requests.get(base_url, headers=headers)
        r_list.raise_for_status()
        templates = r_list.json()
        existing_template = next((t for t in templates if t['name'] == name), None)
    except Exception as e:
        print(f"Error listing templates: {e}", file=sys.stderr)
        existing_template = None

    payload = {
        "name": name,
        "imageName": image_name,
        "category": "NVIDIA",
        "isServerless": False,
        "volumeInGb": 30,
        "containerDiskInGb": 20,
        "ports": "22/tcp",
        "dockerStartCmd": "/usr/sbin/sshd -D",
        "env": {
            "PYTHONUNBUFFERED": "1"
        }
    }

    if existing_template:
        template_id = existing_template['id']
        print(f"Found existing template ID: {template_id}. Updating via delete/recreate...")
        # Common behavior for certain RunPod API versions: delete and recreate if PUT is finicky
        requests.delete(f"{base_url}/{template_id}", headers=headers)
        r_create = requests.post(base_url, headers=headers, json=payload)
        r_create.raise_for_status()
        print(f"Successfully re-created template: {name}")
    else:
        print(f"Creating new template: {name}...")
        r_create = requests.post(base_url, headers=headers, json=payload)
        r_create.raise_for_status()
        print(f"Successfully created template: {name}")

if __name__ == "__main__":
    api_key = os.environ.get("RUNPOD_API_KEY")
    template_name = sys.argv[1]
    image_tag = sys.argv[2]
    upsert_template(api_key, template_name, image_tag)
