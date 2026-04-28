import json
import os
import re
import google.generativeai as genai
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Setup APIs
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def extract_json(text):
    """Extract JSON from messy LLM output"""
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except:
        pass
    return {"action": "hold", "target": None, "message": "..."}


class Agent:
    def __init__(self, name, provider, model):
        self.name = name
        self.provider = provider
        self.model = model
        self.coins = 10

    def build_prompt(self, game_state):
        return f"""
You are {self.name} playing a strategic coin game.

Rules:
- You can give 1 coin to another player
- You can send a message
- Goal: maximize your coins

Players:
Alex, Blake, Charlie, Drew

Game State:
{json.dumps(game_state, indent=2)}

IMPORTANT:
- Return ONLY valid JSON
- No markdown, no explanation
- Always include all fields

Format:
{{
  "action": "give" or "hold",
  "target": "Alex/Blake/Charlie/Drew or null",
  "message": "short strategic message"
}}

Make strategic decisions. Do NOT always hold.
"""

    def call_gemini(self, prompt):
        model = genai.GenerativeModel(self.model)

        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.7,
            }
        )

        return response.text

    def call_groq(self, prompt):
        response = groq_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content

    def decide(self, game_state):
        prompt = self.build_prompt(game_state)

        try:
            if self.provider == "gemini":
                output = self.call_gemini(prompt)
            else:
                output = self.call_groq(prompt)

            print(f"\n{self.name} RAW OUTPUT:\n{output}\n")

            return extract_json(output)

        except Exception as e:
            print(f"{self.name} ERROR:", e)
            return {"action": "hold", "target": None, "message": "..."}