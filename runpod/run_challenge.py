import requests
import json
import os
import sys
import argparse
import time
import traceback

# Configuration
API_KEY = os.environ.get("RUNPOD_API_KEY")

def get_template_by_name(name):
    """Retrieve template ID by name."""
    url = "https://rest.runpod.io/v1/templates"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    try:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        templates = r.json()
        for t in templates:
            if t['name'] == name:
                return t['id']
    except Exception as e:
        print(f"⚠️ Error listing templates: {e}", file=sys.stderr)
    return None

def deploy_pod_rest(gpu_type, count, template_id, volume_id=None, ssh_key=None):
    # Using REST API for deployment as GraphQL schema is unstable
    url = "https://rest.runpod.io/v1/pods"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "name": f"Parameter_Golf_{gpu_type}_{count}x",
        "templateId": template_id,
        "gpuCount": count,
        "gpuTypeIds": [gpu_type], 
        "cloudType": os.environ.get("RUNPOD_CLOUD_TYPE", "SECURE"),
        "env": {}
    }
    if volume_id:
        payload["networkVolumeId"] = volume_id
    if ssh_key:
        payload["env"]["SSH_PUBLIC_KEY"] = ssh_key
    
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"❌ FATAL ERROR: {str(e)}", file=sys.stderr)
        if hasattr(e, 'response') and e.response is not None:
             print(f"Response: {e.response.text}", file=sys.stderr)
        sys.exit(1)

def terminate_pod_rest(pod_id):
    url = f"https://rest.runpod.io/v1/pods/{pod_id}"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    r = requests.delete(url, headers=headers)
    return r.status_code

def main():
    parser = argparse.ArgumentParser(description="RunPod Challenge Deployer (REST)")
    parser.add_argument("--count", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--gpu_type", type=str, default="NVIDIA H100 80GB HBM3", help="GPU Type ID")
    parser.add_argument("--template", type=str, default="Project Golf (H100 Optimized)", help="RunPod template ID OR Name")
    parser.add_argument("--network_volume_id", type=str, help="Network Volume ID")
    parser.add_argument("--ssh_public_key", type=str, help="SSH public key string")
    parser.add_argument("--terminate", type=str, help="Pod ID to terminate")
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    args = parser.parse_args()

    if not API_KEY:
        print("Error: RUNPOD_API_KEY is not set", file=sys.stderr)
        sys.exit(1)

    try:
        if args.terminate:
            status = terminate_pod_rest(args.terminate)
            if args.json: print(json.dumps({"status_code": status}))
            else: print(f"Termination request sent. Status: {status}")
            return

        # 1. Resolve Template ID by name if needed
        template_id = args.template
        # if the template input contains more than a few chars and isn't purely alphanumeric hex, treat as name
        # (RunPod IDs are usually 15-20 chars alpha-numeric)
        if len(template_id) > 10 and " " in template_id:
            print(f"🔍 Searching for template ID for name: {template_id}", file=sys.stderr)
            found_id = get_template_by_name(template_id)
            if found_id:
                print(f"✅ Found template ID: {found_id}", file=sys.stderr)
                template_id = found_id
            else:
                print(f"⚠️ Template name '{template_id}' not found. Using as raw ID...", file=sys.stderr)

        # 2. Deploy
        # We don't specify dataCenterId as per user request to allow 'any' region
        res = deploy_pod_rest(args.gpu_type, args.count, template_id, args.network_volume_id, args.ssh_public_key)
        
        if 'id' not in res:
            if args.json: print(json.dumps(res))
            else: print(f"Deployment Error: {res}")
            sys.exit(1)

        pod_id = res['id']
        output = {"pod_id": pod_id, "status": "REQUESTED"}
        if args.json:
            print(json.dumps(output))
        else:
            print(f"Pod Deployed: {pod_id}")

    except Exception as e:
        print(f"❌ FATAL ERROR: {str(e)}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
