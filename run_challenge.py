import requests
import json
import time
import sys
import argparse

import os

# Configuration
API_KEY = os.environ.get("RUNPOD_API_KEY")
URL = "https://api.runpod.io/graphql"

def run_query(query, variables=None):
    if not API_KEY:
        print("DEBUG: API_KEY is missing!")
    else:
        print(f"DEBUG: API_KEY length: {len(API_KEY)}")
        
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    try:
        response = requests.post(URL, json={'query': query, 'variables': variables}, headers=headers)
        if response.status_code != 200:
            print(f"DEBUG: HTTP Error {response.status_code}: {response.text}")
            return {"errors": [{"message": f"HTTP {response.status_code}: {response.text}"}]}
        return response.json()
    except Exception as e:
        print(f"DEBUG: Exception during request: {e}")
        return {"errors": [{"message": str(e)}]}

def get_gpu_ids(gpu_name="H100"):
    query = "{ gpuTypes { id displayName } }"
    res = run_query(query)
    if 'data' not in res: 
        print(f"Error fetching GPU types: {res}")
        return []
    
    matches = []
    for gpu in res['data']['gpuTypes']:
        if gpu_name in gpu['displayName']:
            # Prioritize SXM
            if "SXM" in gpu['displayName']:
                matches.insert(0, gpu['id'])
            else:
                matches.append(gpu['id'])
    return matches

def deploy_pod(gpu_id, count=1, template_id="t7iu9ugzpi", cloud="SECURE", ssh_public_key=None, network_volume_id=None):
    # Using On-Demand mutation
    mutation = """
    mutation ($input: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $input) {
        id
      }
    }
    """
    input_data = {
        "gpuTypeId": gpu_id,
        "gpuCount": count,
        "cloudType": cloud,
        "templateId": template_id,
        "containerDiskInGb": 50,
        "volumeInGb": 50,
        "startSsh": True,
        "env": []
    }
    if ssh_public_key:
        input_data["env"].append({"key": "PUBLIC_KEY", "value": ssh_public_key})
    if network_volume_id:
        input_data["networkVolumeId"] = network_volume_id
        
    variables = {"input": input_data}
    res = run_query(mutation, variables)
    if 'errors' in res:
        return res
    return res['data']['podFindAndDeployOnDemand']

def terminate_pod(pod_id):
    mutation = """
    mutation ($input: PodTerminateInput!) {
      podTerminate(input: $input)
    }
    """
    variables = {"input": {"podId": pod_id}}
    res = run_query(mutation, variables)
    return res

def wait_for_pod(pod_id):
    query = """
    query ($podId: String!) {
      pod(input: {podId: $podId}) {
        id
        runtime {
          ports {
            ip
            publicPort
            privatePort
          }
        }
      }
    }
    """
    while True:
        res = run_query(query, {"podId": pod_id})
        if 'data' not in res or not res['data']['pod']:
            return None
        pod = res['data']['pod']
        if pod['runtime'] and pod['runtime']['ports']:
            return pod
        time.sleep(10)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--terminate", help="Pod ID to terminate")
    parser.add_argument("--json", action="store_true", help="Output only JSON")
    parser.add_argument("--count", type=int, default=1, help="GPU Count")
    parser.add_argument("--template", default="t7iu9ugzpi", help="Template ID")
    parser.add_argument("--cloud", default=None, help="Cloud Type (COMMUNITY or SECURE). If None, tries both.")
    parser.add_argument("--ssh_public_key", help="Public key string to inject")
    parser.add_argument("--network_volume_id", help="Network volume ID to attach")
    args = parser.parse_args()

    if args.terminate:
        res = terminate_pod(args.terminate)
        if args.json:
            print(json.dumps(res))
        else:
            print(f"Termination requested: {res}")
        return

    gpu_ids = get_gpu_ids()
    if not gpu_ids:
        if args.json:
            print(json.dumps({"error": "No H100 variants found"}))
        else:
            print("No H100 variants found.")
        sys.exit(1)

    # Strategy: Try SECURE then COMMUNITY for each GPU variant
    clouds = [args.cloud] if args.cloud else ["SECURE", "COMMUNITY"]
    
    pod_id = None
    last_error = None
    
    for gpu_id in gpu_ids:
        for cloud in clouds:
            if not args.json:
                print(f"Attempting to deploy {gpu_id} on {cloud} cloud...")
            pod_data = deploy_pod(gpu_id, count=args.count, template_id=args.template, cloud=cloud, ssh_public_key=args.ssh_public_key, network_volume_id=args.network_volume_id)
            
            if pod_data and not (isinstance(pod_data, dict) and 'errors' in pod_data):
                pod_id = pod_data['id']
                break
            else:
                last_error = pod_data
        if pod_id: break

    if not pod_id:
        if args.json:
            print(json.dumps({"error": "Failed to deploy on any H100 variant", "details": last_error}))
        else:
            print(f"Failed to deploy on any H100 variant: {last_error}")
        sys.exit(1)

    pod = wait_for_pod(pod_id)
    if not pod:
        if args.json:
            print(json.dumps({"error": "Pod failed to start", "pod_id": pod_id}))
        else:
            print("Pod failed to start.")
        sys.exit(1)

    ssh_port = None
    ip = None
    for port in pod['runtime']['ports']:
        if port['privatePort'] == 22:
            ssh_port = port['publicPort']
            ip = port['ip']
            break
    
    result = {
        "pod_id": pod_id,
        "ip": ip,
        "port": ssh_port
    }

    if args.json:
        print(json.dumps(result))
    else:
        print(f"Pod deployed: {pod_id}")
        print(f"SSH: ssh root@{ip} -p {ssh_port}")

if __name__ == "__main__":
    main()
