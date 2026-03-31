import requests
import json
import os
import sys
import time
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

def resume_pod(pod_id, gpu_count=1):
    mutation = """
    mutation($input: PodResumeInput!) {
      podResume(input: $input) {
        id
        desiredStatus
      }
    }
    """
    variables = {"input": {"podId": pod_id, "gpuCount": gpu_count}}
    return run_query(mutation, variables)

def stop_pod(pod_id):
    mutation = """
    mutation($input: PodStopInput!) {
      podStop(input: $input) {
        id
        desiredStatus
      }
    }
    """
    variables = {"input": {"podId": pod_id}}
    return run_query(mutation, variables)

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
    return run_query(query, {"podId": pod_id})

def get_all_resources():
    # Only standard fields to ensure validation passes
    query = """
    query {
      myself {
        pods {
          id
          name
          gpuCount
        }
      }
    }
    """
    return run_query(query)

def find_pod(gpu_type="H100", count=8):
    res = get_all_resources()
    if 'data' not in res or 'myself' not in res['data']:
        return None
    
    pods = res['data']['myself']['pods']
    # Filter by name and gpuCount
    for pod in pods:
        if pod['gpuCount'] == count:
            pod_name = pod.get('name', '').upper()
            if not gpu_type or gpu_type.upper() in pod_name:
                return pod
    
    # Generic count match
    for pod in pods:
        if pod['gpuCount'] == count:
            return pod
    return None

def find_volume(target_size=30):
    res = get_all_resources()
    if 'data' not in res or 'myself' not in res['data']:
        return None
    
    volumes = res['data']['myself'].get('networkVolumes', [])
    for vol in volumes:
        if vol['size'] == target_size:
            return vol
    return None

def wait_for_pod(pod_id, timeout=300):
    start_time = time.time()
    while time.time() - start_time < timeout:
        res = get_pod_info(pod_id)
        if 'data' in res and res['data']['pod']:
            runtime = res['data']['pod']['runtime']
            if runtime and runtime['ports']:
                return res['data']['pod']
        time.sleep(10)
    return None

def main():
    parser = argparse.ArgumentParser(description="Manage RunPod pod lifecycle.")
    parser.add_argument("--resume", help="Pod ID to resume")
    parser.add_argument("--stop", help="Pod ID to stop")
    parser.add_argument("--info", help="Pod ID to get IP/SSH Port")
    parser.add_argument("--find-pod", action="store_true", help="Find an existing pod")
    parser.add_argument("--find-volume", action="store_true", help="Find a network volume")
    parser.add_argument("--gpu_count", type=int, default=8, help="GPU count")
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

    elif args.find_volume:
        vol = find_volume()
        if vol:
            if args.json: print(json.dumps(vol))
            else: print(f"Found volume: {vol['id']}")
        else:
            if args.json: print(json.dumps({"error": "No matching volume found"}))
            sys.exit(1)

    elif args.resume:
        pod_id = args.resume
        if not pod_id or pod_id == "null":
            pod = find_pod(count=args.gpu_count)
            if pod: pod_id = pod['id']
            else: sys.exit(1)

        res = resume_pod(pod_id, args.gpu_count)
        if 'errors' in res:
            if args.json: print(json.dumps(res))
            sys.exit(1)
            
        pod = wait_for_pod(pod_id)
        if pod:
            ssh_port, ip = None, None
            for port in pod['runtime']['ports']:
                if port['privatePort'] == 22:
                    ssh_port, ip = port['publicPort'], port['ip']
                    break
            result = {"pod_id": pod_id, "ip": ip, "port": ssh_port, "status": "RUNNING", "was_resumed": True}
            if args.json: print(json.dumps(result))
        else:
            sys.exit(1)

    elif args.stop:
        pod_id = args.stop
        if not pod_id or pod_id == "null":
            pod = find_pod(count=args.gpu_count)
            if pod: pod_id = pod['id']
            else: sys.exit(1)

        res = stop_pod(pod_id)
        if args.json: print(json.dumps(res))

    elif args.info:
        pod = get_pod_info(args.info)
        if args.json: print(json.dumps(pod))

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
