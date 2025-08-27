import requests
import json
import os
from pathlib import Path
from urllib.parse import urljoin

# Base URL for the API
BASE_URL = "https://agents-course-unit4-scoring.hf.space"

def download_resource(question_id: str, file_name: str) -> bool:
    """
    Download a resource file for a specific question.
    
    Args:
        question_id: The ID of the question
        file_name: The name of the file to download
        
    Returns:
        bool: True if download successful, False otherwise
    """
    if not file_name:
        return False
    
    # Create resources directory structure
    resource_dir = Path(f"./data/resources/{question_id}")
    resource_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct download URL
    download_url = urljoin(BASE_URL, f"files/{question_id}")
    file_path = resource_dir / file_name
    
    try:
        print(f"Downloading {file_name} for question {question_id}...")
        response = requests.get(download_url, timeout=30)
        response.raise_for_status()
        
        # Save the file
        with open(file_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✓ Downloaded {file_name} to {file_path}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Failed to download {file_name} for question {question_id}: {e}")
        return False
    except Exception as e:
        print(f"✗ Error saving {file_name} for question {question_id}: {e}")
        return False

def main():
    """Main function to fetch questions and download resources."""
    print("Fetching questions from API...")
    
    # Fetch questions
    questions_url = f"{BASE_URL}/questions"
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions = response.json()
        print(f"✓ Fetched {len(questions)} questions")
    except Exception as e:
        print(f"✗ Failed to fetch questions: {e}")
        return
    
    # Save questions to JSON file
    questions_file = "./data/questions.json"
    os.makedirs("./data", exist_ok=True)
    
    with open(questions_file, "w") as f:
        json.dump(questions, f, indent=2)
    
    print(f"✓ Saved questions to {questions_file}")
    
    # Download resources for questions that have attachments
    print("\nDownloading attached resources...")
    resources_downloaded = 0
    total_resources = 0
    
    for question in questions:
        task_id = question.get("task_id")
        file_name = question.get("file_name", "")
        
        if file_name:
            total_resources += 1
            if download_resource(task_id, file_name):
                resources_downloaded += 1
    
    print(f"\nResource download summary:")
    print(f"Total resources found: {total_resources}")
    print(f"Successfully downloaded: {resources_downloaded}")
    print(f"Failed downloads: {total_resources - resources_downloaded}")
    
    if resources_downloaded > 0:
        print(f"\nResources saved to: ./data/resources/")
        print("Each question's resources are organized in separate folders by question ID.")

if __name__ == "__main__":
    main()



