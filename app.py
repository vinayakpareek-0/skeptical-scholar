import gradio as gr
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from generation.run_generation import run_generation


def format_response(result):
    if result["status"] == "idk":
        response = f"**I don't know.**\n\n"
        response += f"**Reason:** {result['reason']}\n\n"
        if result.get("explanation"):
            response += f"{result['explanation']}"
        if result.get("timings"):
            response += f"\n\n---\n\n**Latency:** {result['timings']['total']:.2f}s"
        return response

    conf = result["confidence"]["score"]
    nli = result["nli"]
    citations = result.get("citations", [])

    response = result["answer"] + "\n\n---\n\n"
    response += f"**Confidence:** {conf:.3f}\n\n"
    if nli.get("skipped"):
        response += "**NLI Verification:** skipped in fast demo mode\n\n"
    else:
        response += f"**NLI Verification:** {nli['supported']*100:.0f}% supported, {nli['contradicted']*100:.0f}% contradicted\n\n"

    if citations:
        response += "**Sources:**\n"
        for c in citations:
            response += f"- `{c['paper_id']}` ({c['section']})\n"

    if result.get("timings"):
        response += f"\n**Latency:** {result['timings']['total']:.2f}s\n"

    return response


def chat(message, history):
    result = run_generation(message)
    return format_response(result)


demo = gr.ChatInterface(
    fn=chat,
    title="Skeptical Scholar",
    description="Ask questions about AI/ML research. Answers are grounded in retrieved ArXiv papers with NLI verification. The system will say 'I don't know' when evidence is insufficient.",
    examples=[
        "What is chain-of-thought prompting?",
        "Explain the attention mechanism in transformers",
        "How does RAG reduce hallucination?",
        "What is the best pizza recipe?",
    ],
)

if __name__ == "__main__":
    demo.launch()
