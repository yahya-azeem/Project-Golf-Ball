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
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    response = requests.post(URL, json={'query': query, 'variables': variables}, headers=headers)
    return response.json()

def get_gpu_id(gpu_name="H100"):
    query = "{ gpuTypes { id displayName } }"
    res = run_query(query)
    if 'data' not in res: return None
    for gpu in res['data']['gpuTypes']:
        if gpu_name in gpu['displayName']:
            return gpu['id']
    return None

def deploy_pod(gpu_id, count=1, template_id="t7iu9ugzpi"):
    # Using On-Demand mutation
    mutation = """
    mutation ($input: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $input) {
        id
      }
    }
    """
    variables = {
        "input": {
            "gpuTypeId": gpu_id,
            "gpuCount": count,
            "cloudType": "COMMUNITY",
            "templateId": template_id,
            "containerDiskInGb": 50,
            "volumeInGb": 50
        }
    }
    res = run_query(mutation, variables)
    if 'errors' in res:
        print(f"Deployment Errors: {res['errors']}")
        return None
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
    args = parser.parse_args()

    if args.terminate:
        res = terminate_pod(args.terminate)
        if not args.json: print(f"Termination requested: {res}")
        else: print(json.dumps(res))
        return

    gpu_id = get_gpu_id()
    if not gpu_id:
        if not args.json: print("H100 not found.")
        return

    pod_data = deploy_pod(gpu_id, count=args.count, template_id=args.template)
    if not pod_data:
        if not args.json: print("Failed to deploy.")
        return

    pod_id = pod_data['id']
    pod = wait_for_pod(pod_id)
    if not pod:
        if not args.json: print("Pod failed to start.")
        return

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
