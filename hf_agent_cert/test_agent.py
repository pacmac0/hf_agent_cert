"""
Quick and dirty test runner: load first 5 questions and run the Agent.
"""

import json
import os
import sys
from typing import List, Dict, Any

from hf_agent_cert.agent import Agent

from dotenv import load_dotenv

# Load environment variables from a .env file if present.
load_dotenv()


def load_first_questions(json_path: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Load the first N questions from a JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[:limit]


def main() -> None:
    """Run the agent on the first five questions and print answers."""
    questions_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "questions.json")

    if not os.getenv("GEMINI_API_KEY"):
        print("GEMINI_API_KEY not set; export it to run the agent.")
        sys.exit(1)

    try:
        questions = load_first_questions(questions_path, limit=15)
        # questions = [questions[9]] # TODO remove after debugging
        questions = questions[0:5] # TODO remove after debugging
    except Exception as e:
        print(f"Failed to load questions: {e}")
        sys.exit(1)

    try:
        agent = Agent(api_key=os.getenv("GEMINI_API_KEY"))
    except Exception as e:
        print(f"Failed to init Agent: {e}")
        sys.exit(1)

    for idx, item in enumerate(questions, start=1):
        task_id = item.get("task_id", f"q{idx}")
        try:
            answer = agent(item)
        except Exception as e:
            answer = f"Error: {e}"
        question_text = item.get("question", "")
        print(f"\n--- Question {idx} ---")
        print(f"Task ID: {task_id}")
        print(f"Question: {question_text}")
        print(f"Answer: {answer}")
        print("-" * 50)


if __name__ == "__main__":
    main()





