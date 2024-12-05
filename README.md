## Installation
- Create a Python virtual environment. For example using venv:
```
python -m venv .llm
source .llm/bin/activate
```
- Install the requirements:
```
pip install -r requirements.txt
```
- Download the .env file and replace the OPENAI API key with your own if so choose: https://drive.google.com/file/d/14_6HYo3S4nQW9PNqjRbHEZG5-EPyNTWR/view?usp=sharing
- Go to the appropriate project directory, for example HED tagging bot:
```
cd hed-bot/hed-tagging-bot
```
- Run the bot in the desired port, for example 55000:
```
chainlit run --port 55000 app.py -w
```