import requests
import json

# URL of the local server's endpoint
URL = "http://127.0.0.1:5000/predict_survey"

# Sample survey data matching the model's expected features
# These values are chosen to be in the middle of the typical 0-4 or 0-5 range.
sample_answers = {
    "blood_pressure": 2,
    "sleep_quality": 2,
    "academic_performance": 2,
    "teacher_student_relationship": 2,
    "basic_needs": 2
}

def test_survey_endpoint():
    """
    Sends a sample request to the /predict_survey endpoint and prints the result.
    """
    print(f"▶️  Sending POST request to {URL}")
    print("--------------------------------------------------")
    print("Request Payload:")
    print(json.dumps({"answers": sample_answers}, indent=2))
    print("--------------------------------------------------")

    try:
        response = requests.post(URL, json={"answers": sample_answers})
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        print("✅ SUCCESS: Received response from server.")
        print("--------------------------------------------------")
        print("Response JSON:")
        print(response.json())
        print("--------------------------------------------------")

    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR: Could not connect to the server at {URL}.")
        print("Please make sure the Flask server in 'app.py' is running.")
        print(f"Error details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    test_survey_endpoint()
