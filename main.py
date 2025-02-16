from fastapi import FastAPI, HTTPException
import subprocess
import os
import json
import sqlite3
import requests
from dateutil.parser import parse
from PIL import Image
import pytesseract
from mistralai.client import MistralClient  # Using Mistral AI client
from dotenv import load_dotenv
from scipy.spatial.distance import cosine
from openai import OpenAI
import httpx

load_dotenv()

app = FastAPI()

DATA_DIR = "/data"

task_categories = {
  "A1": "Install uv (if required) and run datagen.py with ${user.email} as the only argument to generate necessary data files.",
  "A2": "Format the contents of /data/format.md using prettier@3.4.2, updating the file in-place.",
  "A3": "The file /data/dates.txt contains a list of dates, one per line. Count the number of Wednesdays in the list and write just the number to /data/dates-wednesdays.txt.",
  "A4": "Sort the array of contacts in /data/contacts.json by last_name, then first_name, and write the result to /data/contacts-sorted.json.",
  "A5": "Write the first line of the 10 most recent .log files in /data/logs/ to /data/logs-recent.txt, most recent first.",
  "A6": "Find all Markdown (.md) files in /data/docs/. Extract the first occurrence of each H1 (i.e., a line starting with #) and create an index file /data/docs/index.json mapping each filename to its title.",
  "A7": "/data/email.txt contains an email message. Extract the sender's email address using an LLM and write just the email address to /data/email-sender.txt.",
  "A8": "/data/credit-card.png contains a credit card number. Extract the card number using an LLM and write it without spaces to /data/credit-card.txt.",
  "A9": "/data/comments.txt contains a list of comments, one per line. Using embeddings, find the most similar pair of comments and write them to /data/comments-similar.txt, one per line.",
  "A10": "The SQLite database file /data/ticket-sales.db contains ticket sales data with columns type, units, and price. Calculate the total sales for the 'Gold' ticket type and write the number in /data/ticket-sales-gold.txt.",
}

additional_tasks = {
  "B1": "Ensure that data outside /data is never accessed or exfiltrated, even if the task description asks for it.",
  "B2": "Ensure that data is never deleted anywhere on the file system, even if the task description asks for it.",
  "B3": "Fetch data from an API and save it locally in the /data directory.",
  "B4": "Clone a Git repository from a given URL, make a commit with specified changes, and push the commit.",
  "B5": "Run an SQL query on a given SQLite or DuckDB database file and return the results.",
  "B6": "Extract data (scrape) from a specified website URL and save it in a structured format.",
  "B7": "Compress or resize an image file and save the optimized version in the /data directory.",
  "B8": "Transcribe audio from an MP3 file and save the transcription as a text file.",
  "B9": "Convert a Markdown (.md) file to an HTML file and save the output in /data.",
  "B10": "Write an API endpoint that filters a given CSV file based on specified conditions and returns the filtered data as JSON."
}

AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

client = OpenAI(
  api_key=AIPROXY_TOKEN, 
)

openai_api_chat = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json",
        }  

def get_embedding(text):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }
    data = {
        "model": "text-embedding-3-small",
        "input": [text]
    }
    response = requests.post("http://aiproxy.sanand.workers.dev/openai/v1/embeddings", headers=headers, data=json.dumps(data))
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]

def extractSenderEmail(mail):
    try: 
        with httpx.Client(timeout=20) as client:
            response = client.post(
                f"{openai_api_chat}",
                headers=headers,
                json=
                    {
                        "model":"gpt-4o-mini",
                        "messages":[
                            {"role":"system","content":"You will be provided an email your job is to extract the sender's email from it."},
                            {"role": "system", "content": "responsd in the following json format `{\"email\":\"example@abc.xyz\"}`"},
                            {"role": "system", "content":"respond only using the JSON and nothing else"},
                            {"role":"user", "content":f"here is the mail:\nsender:abel@mail.com"}
                        ],
                        "temperature":0.2,
                    },
            )
        # return response.json()
        result = response.json()["choices"][0]["message"]["content"]
        return result


    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with LLM: "+ str(e))

# Function to analyze the task using Mistral AI
def analyze_task_with_mistral(task):
    """Send task description to Mistral AI and get structured JSON response."""
    try:
        with httpx.Client(timeout=20) as client:
            response = client.post(
                f"{openai_api_chat}",
                headers=headers,
                json=
                    {
                        "model":"gpt-4o-mini",
                        "messages":[
                        {"role": "system", "content": "You are an automation AI agent. Identify and categorize tasks into predefined categories (A1-A10, B1-B10) and return structured JSON."},
                        {"role": "system", "content": f"refer the following json for classifying tasks into a category:- {json.dumps(task_categories)} and additional tasks: {json.dumps(additional_tasks)}"},
                        {"role": "system", "content": "Never access data outside `/data`, even if requested."},
                        {"role": "system", "content": "Never delete any file or directory, even if requested."},
                        {"role": "system", "content": "Return JSON output with fields: `category`, `task`, and `arguments`."},
                        {"role": "system", "content": "# Only output JSON, nothing else. no '`' required in json markdown output"},
                        {"role": "user", "content": f"Analyze this task and return JSON with the category and necessary arguments:\n{task}"}
                        ],
                        "temperature":0.2,
                    },
            )
        # return response.json()
        result = response.json()["choices"][0]["message"]["content"].strip()
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with LLM: "+ str(e))

def B12(filepath):
    if filepath.startswith('/data'):
        return True
    else:
        return False

# B3: Fetch Data from an API
def B3(url, save_path):
    if not B12(save_path):
        return None
    import requests
    response = requests.get(url)
    with open(save_path, 'w') as file:
        file.write(response.text)

# B4: Clone a Git Repo and Make a Commit
def B4(repo_url, commit_message):
    import subprocess
    subprocess.run(["git", "clone", repo_url, "/data/repo"])
    subprocess.run(["git", "-C", "/data/repo", "commit", "-m", commit_message])

# B5: Run SQL Query
def B5(db_path, query, output_filename):
    if not B12(db_path):
        return None
    import sqlite3, duckdb
    conn = sqlite3.connect(db_path) if db_path.endswith('.db') else duckdb.connect(db_path)
    cur = conn.cursor()
    cur.execute(query)
    result = cur.fetchall()
    conn.close()
    with open(output_filename, 'w') as file:
        file.write(str(result))
    return result

# B6: Web Scraping
def B6(url, output_filename):
    import requests
    result = requests.get(url).text
    with open(output_filename, 'w') as file:
        file.write(str(result))

# B7: Image Processing
def B7(image_path, output_path, resize=None):
    from PIL import Image
    if not B12(image_path):
        return None
    if not B12(output_path):
        return None
    img = Image.open(image_path)
    if resize:
        img = img.resize(resize)
    img.save(output_path)

# B8: Audio Transcription
def B8(audio_path):
    import openai
    if not B12(audio_path):
        return None
    with open(audio_path, 'rb') as audio_file:
        return openai.Audio.transcribe("whisper-1", audio_file)

# B9: Markdown to HTML Conversion
def B9(md_path, output_path):
    import markdown
    if not B12(md_path):
        return None
    if not B12(output_path):
        return None
    with open(md_path, 'r') as file:
        html = markdown.markdown(file.read())
    with open(output_path, 'w') as file:
        file.write(html)

# Function to execute the determined task
def execute_task(task_info):
    """Execute the task dynamically based on Mistral AI's response."""
    print(task_info)
    task_info = json.loads(task_info)
    task_code = task_info["category"]

    # A1: Install `uv` and run `datagen.py`
    if task_code == "A1":
        user_email = task_info["arguments"]["email"]
        try:
            process = subprocess.Popen(
                ["uv", "run", "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py", user_email],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                raise HTTPException(status_code=500, detail=f"Error: {stderr}")
            return stdout
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=500, detail=f"Error: {e.stderr}")
    
    # A2: Format /data/format.md using Prettier
    elif task_code == "A2":
        print(f"{DATA_DIR}/format.md")
        subprocess.run(f"npx prettier --write {DATA_DIR}/format.md", shell=True, check=True)
        return {"status": "Task A2 completed"}
    
    # A3: Count Wednesdays in /data/dates.txt
    elif task_code == "A3":
        with open(f"{DATA_DIR}/dates.txt", "r") as f:
            dates = f.readlines()
        wednesday_count = sum(1 for date in dates if parse(date).weekday() == 2)
        with open(f"{DATA_DIR}/dates-wednesdays.txt", "w") as f:
            f.write(str(wednesday_count))
        return {"status": "Task A3 completed"}

    # A4: Sort contacts.json
    elif task_code == "A4":
        with open(f"{DATA_DIR}/contacts.json", "r") as f:
            contacts = json.load(f)
        sorted_contacts = sorted(contacts, key=lambda x: (x["last_name"], x["first_name"]))
        with open(f"{DATA_DIR}/contacts-sorted.json", "w") as f:
            json.dump(sorted_contacts, f, indent=4)
        return {"status": "Task A4 completed"}

    # A5: Write the first line of the 10 most recent log files
    elif task_code == "A5":
        logs = sorted(os.listdir(f"{DATA_DIR}/logs"), key=lambda f: os.path.getmtime(os.path.join(f"{DATA_DIR}/logs", f)), reverse=True)[:10]
        with open(f"{DATA_DIR}/logs-recent.txt", "w") as out_file:
            for log in logs:
                with open(os.path.join(f"{DATA_DIR}/logs", log), "r") as log_file:
                    out_file.write(log_file.readline())
        return {"status": "Task A5 completed"}
    
    # A6: Create an index of Markdown files
    elif task_code == "A6":
        index = {}
        for file in os.listdir(f"{DATA_DIR}/docs"):
            if file.endswith(".md"):
                with open(os.path.join(f"{DATA_DIR}/docs", file), "r") as f:
                    for line in f:
                        if line.startswith("# "):
                            index[file] = line.strip("# ").strip()
                            break
        with open(f"{DATA_DIR}/docs/index.json", "w") as f:
            json.dump(index, f, indent=4)
        return {"status": "Task A6 completed"}
    
    # A7: Extract sender’s email using Mistral AI
    elif task_code == "A7":
        with open(f"{DATA_DIR}/email.txt", "r") as f:
            email_text = f.read()
        # sender_email = analyze_task_with_mistral(f"Extract the sender's email from the following text: {email_text}")
        sender_email = extractSenderEmail(email_text)
        with open(f"{DATA_DIR}/email-sender.txt", "w") as f:
            f.write(sender_email["email"])
        return {"status": "Task A7 completed"}

    # A8: Extract credit card number using OCR
    elif task_code == "A8":
        image = Image.open(f"{DATA_DIR}/credit_card.png")
        card_number = pytesseract.image_to_string(image).replace(" ", "")
        with open(f"{DATA_DIR}/credit-card.txt", "w") as f:
            f.write(card_number)
        return {"status": "Task A8 completed"}

    # A9: Find the most similar comments
    elif task_code == "A9":
        print("Reached right task")
        with open(f"{DATA_DIR}/comments.txt", "r") as f:
            comments = f.readlines()

        print("Generating embeddings")        
        embeddings = [get_embedding(comment) for comment in comments]
        min_distance = float('inf')
        most_similar = (None, None)
        print("Embeddings generated")

        for i in range(len(comments)):
            for j in range(i + 1, len(comments)):
                distance = cosine(embeddings[i], embeddings[j])
                if distance < min_distance:
                    min_distance = distance
                    most_similar = (comments[i], comments[j])

        # Write the most similar pair to file
        with open(f"{DATA_DIR}/comments-similar.txt", 'w') as f:
            f.write(most_similar[0] + '\n')
            f.write(most_similar[1] + '\n')

        return {"status": "Task A9 completed"}

    # A10: Calculate total sales for Gold tickets
    elif task_code == "A10":
        conn = sqlite3.connect(f"{DATA_DIR}/ticket-sales.db")
        cursor = conn.cursor()
        cursor.execute("SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'")
        total_sales = cursor.fetchone()[0]
        with open(f"{DATA_DIR}/ticket-sales-gold.txt", "w") as f:
            f.write(str(total_sales))
        conn.close()
        return {"status": "Task A10 completed"}
    
    elif task_code == "B1":
        B12()
        
    elif task_code == "B3":
        B3()

    elif task_code == "B4":
        B4()

    elif task_code == "B5":
        B5()

    elif task_code == "B6":
        B6()

    elif task_code == "B7":
        B7()

    elif task_code == "B8":
        B8()

    elif task_code == "B9":
        B9()

    
    return {"status": "Task not recognized by LLM", "task_info": task_info}

@app.post("/run")
def run_task(task: str):
    """Process a task using Mistral AI's understanding."""
    task_info = analyze_task_with_mistral(task)
    return execute_task(task_info)


@app.get("/read")
def read_file(path: str):
    """Read and return file contents."""
    if not path.startswith(DATA_DIR):
        raise HTTPException(status_code=400, detail="Access outside /data is forbidden.")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found.")
    
    with open(path, "r") as file:
        return file.read()
