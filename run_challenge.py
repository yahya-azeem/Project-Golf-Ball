import requests
import json
import os
import sys
import argparse
import time

# Configuration
API_KEY = os.environ.get("RUNPOD_API_KEY")

def deploy_pod_rest(gpu_type, count, template_id, volume_id=None, ssh_key=None):
    # Using REST API for deployment as GraphQL schema is unstable
    url = "https://rest.runpod.io/v1/pods"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "name": f"Parameter_Golf_{gpu_type}_{count}x",
        "imageName": None, # Provided by template
        "templateId": template_id,
        "gpuCount": count,
        "gpuTypeId": gpu_type,
        "cloudType": "SECURE",
    }
    if volume_id:
        payload["networkVolumeId"] = volume_id
    if ssh_key:
        # Note: some documentation says 'sshPublicKey', others 'publicKey'
        payload["sshPublicKey"] = ssh_key
    
    r = requests.post(url, json=payload, headers=headers)
    return r.json()

def terminate_pod_rest(pod_id):
    url = f"https://rest.runpod.io/v1/pods/{pod_id}/terminate"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    r = requests.post(url, headers=headers)
    return r.status_code

def main():
    parser = argparse.ArgumentParser(description="RunPod Challenge Deployer (REST)")
    parser.add_argument("--count", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--template", type=str, default="t7iu9ugzpi", help="RunPod template ID")
    parser.add_argument("--network_volume_id", type=str, help="Network Volume ID")
    parser.add_argument("--ssh_public_key", type=str, help="SSH public key string")
    parser.add_argument("--terminate", type=str, help="Pod ID to terminate")
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    args = parser.parse_args()

    if not API_KEY:
        print("Error: RUNPOD_API_KEY is not set")
        sys.exit(1)

    if args.terminate:
        status = terminate_pod_rest(args.terminate)
        if args.json: print(json.dumps({"status_code": status}))
        else: print(f"Termination request sent. Status: {status}")
        return

    # In REST, the ID for H100 is often 'NVIDIA H100 80GB HBM3'
    res = deploy_pod_rest("NVIDIA H100 80GB HBM3", args.count, args.template, args.network_volume_id, args.ssh_public_key)
    
    if 'id' not in res:
        if args.json: print(json.dumps(res))
        else: print(f"Deployment Error: {res}")
        sys.exit(1)

    pod_id = res['id']
    
    # In REST response, we get IP/Port directly sometimes, or need to query
    # We will output what we have and let the GHA manage_pod.py handles the discovery
    output = {"pod_id": pod_id, "status": "REQUESTED"}
    if args.json:
        print(json.dumps(output))
    else:
        print(f"Pod Deployed: {pod_id}")

if __name__ == "__main__":
    main()
