import requests



# ============================================================
# Configuration
# ============================================================

JUDGE_URL = "http://192.168.10.54:8000/judge"


# ============================================================
# Send prompt to judge
# ============================================================

def send_to_judge(judge_prompt):

    response = requests.post(
        JUDGE_URL,
        json={
            "prompt": judge_prompt
        }
    )

    response.raise_for_status()

    result = response.json()

    return result["score_1"], result["score_2"]