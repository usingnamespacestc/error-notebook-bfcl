import requests
import json

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma4:26b"

tools = [{
    "type": "function",
    "function": {
        "name": "Music_3_PlayMedia",
        "description": "Plays a specified track on a designated media player device.",
        "parameters": {
            "type": "object",
            "required": ["track"],
            "properties": {
                "track": {"type": "string"},
                "artist": {"type": "string"},
                "device": {"type": "string", "enum": ["Living room", "Kitchen", "Patio"]}
            }
        }
    }
}]

messages = [
    {"role": "user", "content": "Can you play the track 'Shape of You' by Ed Sheeran on the kitchen speaker?"}
]

payload = {
    "model": MODEL,
    "messages": messages,
    "tools": tools,
    "stream": False
}

print("Requesting...")
response = requests.post(OLLAMA_URL, json=payload)
print(json.dumps(response.json(), indent=2))
