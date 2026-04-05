import requests
import json
import os
import sys
import time
import socket
import argparse
import traceback

# Configuration
API_KEY = os.environ.get("RUNPOD_API_KEY")
REST_URL = "https://rest.runpod.io/v1/pods"

def get_pod_info(pod_id):
    """Retrieve detailed pod info via REST API."""
    url = f"{REST_URL}/{pod_id}"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    try:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"⚠️ Error fetching pod info for {pod_id}: {e}", file=sys.stderr)
        return None

def find_pod(count=1):
    """Find a running pod with the specified GPU count."""
    headers = {"Authorization": f"Bearer {API_KEY}"}
    try:
        r = requests.get(REST_URL, headers=headers, timeout=30)
        r.raise_for_status()
        pods = r.json()
        for pod in pods:
            if pod.get('gpuCount') == count and pod.get('runtime'):
                return pod
    except Exception as e:
        print(f"⚠️ Error listing pods: {e}", file=sys.stderr)
    return None

def wait_for_pod(pod_id, timeout=600):
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

def extract_ssh_info(pod):
    """Extricate IP and Port for SSH from the pod runtime info."""
    if not pod or 'runtime' not in pod:
        return None, None
    ports = pod['runtime'].get('ports', [])
    for p in ports:
        if p.get('privatePort') == 22 and p.get('publicPort'):
            return p.get('ip'), p.get('publicPort')
    return None, None

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
                print(f"✅ SSH port {port} is open and reachable", file=sys.stderr)
                return True
        except Exception:
            pass
        time.sleep(5)
    return False

def resume_pod_rest(pod_id):
    url = f"{REST_URL}/{pod_id}/start"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    try:
        r = requests.post(url, headers=headers, timeout=30)
        # 400 is often returned if the pod is already running
        if r.status_code != 400:
            r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"⚠️ Warning during pod start: {e}", file=sys.stderr)
        return {}

def main():
    parser = argparse.ArgumentParser(description="RunPod lifecycle manager")
    parser.add_argument("--find-pod", action="store_true")
    parser.add_argument("--resume", type=str, help="Pod ID to resume")
    parser.add_argument("--wait", type=str, help="Pod ID to wait for")
    parser.add_argument("--gpu_count", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not API_KEY:
        print("Error: RUNPOD_API_KEY is not set", file=sys.stderr)
        sys.exit(1)

    try:
        if args.find_pod:
            pod = find_pod(count=args.gpu_count)
            if pod:
                if args.json: print(json.dumps(pod))
                else: print(f"Found pod: {pod['id']}")
            else:
                if args.json: print(json.dumps({"error": "No matching pod found"}))
                sys.exit(1)

        elif args.wait:
            pod_id = args.wait
            pod = wait_for_pod(pod_id)
            if not pod:
                print(json.dumps({"error": "Pod never became ready (API timeout)"}), file=sys.stdout)
                sys.exit(1)

            ip, ssh_port = extract_ssh_info(pod)
            if not ip or not ssh_port:
                print(json.dumps({"error": "SSH port mapping not found in pod info"}), file=sys.stdout)
                sys.exit(1)

            result = {"pod_id": pod_id, "ip": ip, "port": ssh_port, "status": "RUNNING"}
            if args.json: print(json.dumps(result))
            else: print(f"Pod {pod_id} is fully SSH-ready at {ip}:{ssh_port}")

        elif args.resume:
            pod_id = args.resume
            resume_pod_rest(pod_id)
            pod = wait_for_pod(pod_id)
            if not pod:
                sys.exit(1)

            ip, ssh_port = extract_ssh_info(pod)
            if not ip or not ssh_port:
                print("Error: SSH port mapping not found after resume", file=sys.stderr)
                sys.exit(1)

            # Wait for socket-level readiness (shorter timeout)
            wait_for_ssh(ip, ssh_port)

            result = {"pod_id": pod_id, "ip": ip, "port": ssh_port, "status": "RUNNING", "was_resumed": True}
            if args.json: print(json.dumps(result))
            else: print(f"Pod {pod_id} is RUNNING at {ip}:{ssh_port}")

    except Exception as e:
        print(f"❌ FATAL ERROR: {str(e)}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
