import requests
import json
import os
import sys
import time
import argparse

# Configuration
API_KEY = os.environ.get("RUNPOD_API_KEY")
GRAPHQL_URL = "https://api.runpod.io/graphql"
REST_URL = "https://rest.runpod.io/v1/pods"

def run_graphql_query(query, variables=None):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    try:
        response = requests.post(GRAPHQL_URL, json={'query': query, 'variables': variables}, headers=headers, timeout=30)
        return response.json()
    except Exception as e:
        return {"errors": [{"message": str(e)}]}

def resume_pod_rest(pod_id):
    url = f"{REST_URL}/{pod_id}/start"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    r = requests.post(url, headers=headers)
    return r.json()

def stop_pod_rest(pod_id):
    url = f"{REST_URL}/{pod_id}/stop"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    r = requests.post(url, headers=headers)
    return r.json()

def get_pod_info(pod_id):
    query = """
    query ($podId: String!) {
      pod(input: {podId: $podId}) {
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
    res = run_graphql_query(query, {"podId": pod_id})
    if 'data' in res and res['data'].get('pod'):
        return res['data']['pod']
    return None

def get_all_pods_rest():
    headers = {"Authorization": f"Bearer {API_KEY}"}
    r = requests.get(REST_URL, headers=headers)
    return r.json()

def find_pod(count=8):
    pods = get_all_pods_rest()
    if not isinstance(pods, list): return None
    
    # Filter by gpuCount
    for pod in pods:
        if pod.get('gpuCount') == count:
            return pod
    return None

def wait_for_pod(pod_id, timeout=300):
    start_time = time.time()
    while time.time() - start_time < timeout:
        pod = get_pod_info_rest(pod_id)
        if isinstance(pod, dict) and pod.get('runtime'):
            ports = pod['runtime'].get('ports')
            if ports:
                return pod
        time.sleep(10)
    return None

def main():
    parser = argparse.ArgumentParser(description="Manage RunPod pod lifecycle (REST).")
    parser.add_argument("--resume", help="Pod ID to resume")
    parser.add_argument("--stop", help="Pod ID to stop")
    parser.add_argument("--info", help="Pod ID to get IP/SSH Port")
    parser.add_argument("--find-pod", action="store_true", help="Find an existing pod")
    parser.add_argument("--gpu_count", type=int, default=1, help="GPU count")
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    args = parser.parse_args()

    if not API_KEY:
        sys.exit(1)

    if args.find_pod:
        pod = find_pod(count=args.gpu_count)
        if pod:
            if args.json: print(json.dumps(pod))
            else: print(f"Found pod: {pod['id']}")
        else:
            if args.json: print(json.dumps({"error": "No matching pod found"}))
            sys.exit(1)

    elif args.resume:
        pod_id = args.resume
        # First ensure it's started
        resume_pod_rest(pod_id)
        
        # Wait for IP/Port
        pod = wait_for_pod(pod_id)
        if pod:
            ssh_port, ip = None, None
            for port in pod['runtime'].get('ports', []):
                if port.get('privatePort') == 22:
                    ssh_port, ip = port.get('publicPort'), port.get('ip')
                    break
            result = {"pod_id": pod_id, "ip": ip, "port": ssh_port, "status": "RUNNING", "was_resumed": True}
            if args.json: print(json.dumps(result))
            else: print(f"Pod {pod_id} is RUNNING at {ip}:{ssh_port}")
        else:
            sys.exit(1)

    elif args.stop:
        res = stop_pod_rest(args.stop)
        if args.json: print(json.dumps(res))
        else: print(f"Stop request sent for {args.stop}")

    elif args.info:
        pod = get_pod_info_rest(args.info)
        if args.json: print(json.dumps(pod))
        else: print(f"Pod Info: {pod}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
