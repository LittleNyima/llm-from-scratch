import os

import requests


def fetch_example_text(url: str, file_path: str):
    if not os.path.exists(file_path):
        response = requests.get(url)
        response.raise_for_status()
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write(response.text)


if __name__ == "__main__":
    url = "https://static.hoshinorubii.icu/misc/a-tale-of-two-cities.txt"
    file_path = "downloads/data/a-tale-of-two-cities.txt"

    fetch_example_text(url, file_path)
