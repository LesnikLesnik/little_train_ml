import os
import requests

MODEL_PATH = "ml/model/student_depression_model.pkl"
YANDEX_DISK_PUBLIC_URL = "https://disk.yandex.ru/d/IonIz-UDdiYltw"

def get_yandex_direct_download(public_url):
    api_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download"
    params = {"public_key": public_url}
    response = requests.get(api_url, params=params)
    response.raise_for_status()
    return response.json()["href"]


def download_model_if_missing():
    if os.path.exists(MODEL_PATH):
        return True

    print("[INFO] Model not found. Downloading...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    try:
        download_url = get_yandex_direct_download(YANDEX_DISK_PUBLIC_URL)
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("[INFO] Model downloaded successfully.")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download model: {e}")
        return False
