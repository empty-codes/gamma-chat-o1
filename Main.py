from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You're a helpful assistant"
        },
        {
            "role": "user",
            "content": "Tell me something fascinating about big cats"
        }
    ],
    model="llama-3.3-70b-versatile",
    stream=False
)
print(chat_completion.choices[0].message.content)
