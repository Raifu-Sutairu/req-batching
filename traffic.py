import requests
import time
import concurrent.futures

def send_request():
    try:
        requests.post("http://localhost:8080/api/v1/resource", json={"data": "test"}, timeout=2)
    except Exception:
        pass

if __name__ == "__main__":
    print("Sending traffic...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        for _ in range(200):
            for _ in range(50):
                executor.submit(send_request)
            time.sleep(0.05)
    print("Done sending traffic!")
