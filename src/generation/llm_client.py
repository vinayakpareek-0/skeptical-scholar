from dotenv import load_dotenv
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from generation.prompts import build_prompt , build_idk_prompt
from reasoning.run_reasoning import run_reasoning
load_dotenv()
from groq import Groq


def load_llm():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate(client , prompt , model="llama-3.1-8b-instant", temperature = 0.3):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature= temperature,
        max_tokens =512
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    result = run_reasoning("How does retrieval augmented generation reduce hallucination?")

    if result["status"] == "idk":
        print(f"IDK: {result['reason']}")
    else:
        client = load_llm()
        prompt = build_prompt(
            "How does retrieval augmented generation reduce hallucination?",
            result["chunks"],
            result["confidence"]
        )
        answer = generate(client, prompt)
        print(f"Confidence: {result['confidence']['score']}")
        print(f"\nAnswer:\n{answer}")
