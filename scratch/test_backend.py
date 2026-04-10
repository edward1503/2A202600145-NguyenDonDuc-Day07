import requests
import json

BASE_URL = "http://127.0.0.1:8501"

def test_status():
    try:
        response = requests.get(f"{BASE_URL}/api/status")
        print(f"Status: {response.status_code}")
        print(f"Body: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

def test_build():
    try:
        data = {
            "compare_mode": "false",
            "use_sample_data": "true",
            "method_a": "fixed_size",
            "a_params": json.dumps({"chunk_size": 500, "overlap": 50}),
            "method_b": "recursive",
            "b_params": json.dumps("{}")
        }
        print("Sending build request...")
        response = requests.post(f"{BASE_URL}/api/build", data=data, timeout=60)
        print(f"Build: {response.status_code}")
        print(f"Body: {response.json()}")
    except requests.exceptions.Timeout:
        print("Build request timed out (expected if it's slow)")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_status()
    test_build()
