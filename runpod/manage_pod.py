import requests
import json
import os
import sys
import time
import socket
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

def terminate_pod_rest(pod_id):
    url = f"{REST_URL}/{pod_id}"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    r = requests.delete(url, headers=headers)
    # REST API for DELETE usually returns status 204 or a small JSON
    try:
        return r.json()
    except:
        return {"status": r.status_code}

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
    """Wait for RunPod API to report the pod as running with SSH port mapped."""
    start_time = time.time()
    print(f"⏳ Waiting for pod {pod_id} to report SSH readiness via API (timeout={timeout}s)...", file=sys.stderr)
    while time.time() - start_time < timeout:
        pod = get_pod_info(pod_id)
        if isinstance(pod, dict) and pod.get('runtime'):
            ports = pod['runtime'].get('ports', [])
            # specifically check for privatePort 22 mapping WITH truthy publicPort and ip
            if any(p.get('privatePort') == 22 and p.get('publicPort') and p.get('ip') for p in ports):
                elapsed = int(time.time() - start_time)
                print(f"✅ Pod API reports SSH port mapped after {elapsed}s", file=sys.stderr)
                return pod
        time.sleep(10)
    print(f"❌ Timed out waiting for pod {pod_id} API readiness after {timeout}s", file=sys.stderr)
    return None


def wait_for_ssh(ip, port, timeout=120):
    """Wait for the SSH service to actually accept TCP connections."""
    start_time = time.time()
    print(f"⏳ Waiting for SSH service at {ip}:{port} to accept connections (timeout={timeout}s)...", file=sys.stderr)
    while time.time() - start_time < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((ip, int(port)))
            sock.close()
            if result == 0:
                elapsed = int(time.time() - start_time)
                print(f"✅ SSH service is accepting connections after {elapsed}s", file=sys.stderr)
                return True
        except (socket.error, OSError) as e:
            pass
        time.sleep(5)
    print(f"❌ Timed out waiting for SSH at {ip}:{port} after {timeout}s", file=sys.stderr)
    return False


def extract_ssh_info(pod):
    """Extract SSH IP and port from pod info dict."""
    for port_info in pod.get('runtime', {}).get('ports', []):
        if port_info.get('privatePort') == 22:
            return port_info.get('ip'), port_info.get('publicPort')
    return None, None

def main():
    parser = argparse.ArgumentParser(description="Manage RunPod pod lifecycle (REST).")
    parser.add_argument("--resume", help="Pod ID to resume")
    parser.add_argument("--stop", help="Pod ID to stop")
    parser.add_argument("--terminate", help="Pod ID to terminate")
    parser.add_argument("--info", help="Pod ID to get IP/SSH Port")
    parser.add_argument("--wait", help="Pod ID to wait for full SSH readiness (API + socket)")
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

    elif args.wait:
        # Full readiness wait: API reports running + SSH port is reachable
        pod_id = args.wait
        pod = wait_for_pod(pod_id, timeout=300)
        if not pod:
            print(json.dumps({"error": "Pod never became ready (API timeout)"}), file=sys.stdout)
            sys.exit(1)

        ip, ssh_port = extract_ssh_info(pod)
        if not ip or not ssh_port:
            print(json.dumps({"error": "SSH port mapping not found in pod info"}), file=sys.stdout)
            sys.exit(1)

        if not wait_for_ssh(ip, ssh_port, timeout=120):
            print(json.dumps({"error": f"SSH service at {ip}:{ssh_port} never became reachable"}), file=sys.stdout)
            sys.exit(1)

        result = {"pod_id": pod_id, "ip": ip, "port": ssh_port, "status": "RUNNING"}
        if args.json: print(json.dumps(result))
        else: print(f"Pod {pod_id} is fully SSH-ready at {ip}:{ssh_port}")

    elif args.resume:
        pod_id = args.resume
        # First ensure it's started
        resume_pod_rest(pod_id)

        # Wait for IP/Port via API
        pod = wait_for_pod(pod_id)
        if not pod:
            sys.exit(1)

        ip, ssh_port = extract_ssh_info(pod)
        if not ip or not ssh_port:
            print("Error: SSH port mapping not found after resume", file=sys.stderr)
            sys.exit(1)

        # Also wait for SSH service to be reachable
        if not wait_for_ssh(ip, ssh_port, timeout=120):
            print(f"Warning: SSH at {ip}:{ssh_port} may not be ready", file=sys.stderr)

        result = {"pod_id": pod_id, "ip": ip, "port": ssh_port, "status": "RUNNING", "was_resumed": True}
        if args.json: print(json.dumps(result))
        else: print(f"Pod {pod_id} is RUNNING at {ip}:{ssh_port}")

    elif args.stop:
        res = stop_pod_rest(args.stop)
        if args.json: print(json.dumps(res))
        else: print(f"Stop request sent for {args.stop}")

    elif args.terminate:
        res = terminate_pod_rest(args.terminate)
        if args.json: print(json.dumps(res))
        else: print(f"Terminate request sent for {args.terminate}")

    elif args.info:
        pod = get_pod_info(args.info)
        if args.json: print(json.dumps(pod))
        else: print(f"Pod Info: {pod}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
