import requests, json, os

API_KEY = os.environ.get("RUNPOD_API_KEY")
POD_ID = "gwqmz96a9fg3y8"

# Resume the pod
mutation = """
mutation($input: PodResumeInput!) {
  podResume(input: $input) {
    id
    desiredStatus
  }
}
"""
variables = {"input": {"podId": POD_ID, "gpuCount": 1}}
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
r = requests.post("https://api.runpod.io/graphql", json={"query": mutation, "variables": variables}, headers=headers)
print(json.dumps(r.json(), indent=2))
