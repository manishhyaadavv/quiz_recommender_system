import json
import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("QUIZ_API_KEY")

if not API_KEY:
    print("API KEY NOT FOUND! Check your .env file.")
    exit()

url = "https://quizapi.io/api/v1/questions"

params = {
    "category": "Programming",
    "tags": "javascript",
    "type": "MULTIPLE_CHOICE",
    "limit": 10,
    "random": "true"
}

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

try:
    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=20
    )

    print("Status Code:", response.status_code)

    result = response.json()

    if response.status_code == 200:
        questions = result.get("data", [])
        if questions:
            print(json.dumps(questions[0], indent=4))

        print("Questions Found:", len(questions))

        for i, question in enumerate(questions, 1):
            print(f"\nQuestion {i}:")
            print(question.get("text"))

            for option in question.get("answers", []):
                print("-", option.get("text"))

    else:
        print("API Error:", result)

except requests.exceptions.RequestException as error:
    print("Connection Error:", error)