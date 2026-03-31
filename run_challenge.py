import requests
import json
import os
import sys
import argparse

# Configuration
API_KEY = os.environ.get("RUNPOD_API_KEY")
URL = "https://api.runpod.io/graphql"

def run_query(query, variables=None):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    try:
        response = requests.post(URL, json={'query': query, 'variables': variables}, headers=headers, timeout=30)
        if response.status_code != 200:
            return {"errors": [{"message": f"HTTP {response.status_code}: {response.text}"}]}
        return response.json()
    except Exception as e:
        return {"errors": [{"message": str(e)}]}

def get_gpu_ids(gpu_name="H100"):
    query = "{ gpuTypes { id displayName } }"
    res = run_query(query)
    if 'data' in res:
        for g in res['data']['gpuTypes']:
            if gpu_name.upper() in g['displayName'].upper():
                return g['id']
    return None

def deploy_pod(gpu_id, count, template_id, volume_id=None, ssh_key=None):
    mutation = """
    mutation ($input: PodCreateInput!) {
      podFindAndDeployOnDemand(input: $input) {
        id
        desiredStatus
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
    input_data = {
        "gpuTypeId": gpu_id,
        "gpuCount": count,
        "templateId": template_id,
    }
    if volume_id:
        input_data["networkVolumeId"] = volume_id
    if ssh_key:
        input_data["publicKey"] = ssh_key
        
    variables = {"input": input_data}
    return run_query(mutation, variables)

def terminate_pod(pod_id):
    mutation = """
    mutation ($input: PodTerminateInput!) {
      podTerminate(input: $input)
    }
    """
    variables = {"input": {"podId": pod_id}}
    return run_query(mutation, variables)

def main():
    parser = argparse.ArgumentParser(description="RunPod Challenge Deployer")
    parser.add_argument("--count", type=int, default=8, help="Number of GPUs")
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
        res = terminate_pod(args.terminate)
        if args.json: print(json.dumps(res))
        else: print(f"Termination result: {res}")
        return

    gpu_id = get_gpu_ids("H100")
    if not gpu_id:
        print("Error: Could not find H100 GPU ID")
        sys.exit(1)

    res = deploy_pod(gpu_id, args.count, args.template, args.network_volume_id, args.ssh_public_key)
    
    if 'errors' in res:
        if args.json: print(json.dumps(res))
        else: print(f"Deployment Error: {res['errors']}")
        sys.exit(1)

    pod = res['data']['podFindAndDeployOnDemand']
    pod_id = pod['id']
    
    # Wait for IP/Port (simplified wait here or handle in GHA)
    ssh_port, ip = None, None
    if pod['runtime'] and pod['runtime']['ports']:
        for port in pod['runtime']['ports']:
            if port['privatePort'] == 22:
                ssh_port, ip = port['publicPort'], port['ip']
                break
    
    output = {"pod_id": pod_id, "ip": ip, "port": ssh_port, "status": "BOOTING" if pod['runtime'] and not ip else "RUNNING" if ip else "REQUESTED"}
    if args.json:
        print(json.dumps(output))
    else:
        print(f"Pod Deployed: {pod_id} at {ip}:{ssh_port}")

if __name__ == "__main__":
    main()
