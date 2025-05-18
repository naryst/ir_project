import requests

def refine_query_for_clip(user_query: str) -> str:
    # LLM-инструкция: переформулировать в визуально насыщенный описательный запрос
    system_prompt = (
        "You are an expert in transforming user queries into visually descriptive prompts "
        "that match the CLIP embedding space for image retrieval tasks.\n\n"
        "Your job is to rewrite the user's query as a vivid, descriptive caption that would align well with an image.\n"
        "Focus on objects, actions, colors, environments, and concrete visual elements.\n"
        "Avoid vague concepts. Stick to the original meaning.\n"
        "Return only the refined prompt, no other text.\n"
        "Here are examples:\n\n"
        "User: 'a car'\n→ 'a red sports car speeding down a highway under a blue sky'\n\n"
        "User: 'dog'\n→ 'a golden retriever playing fetch on a sunny beach'\n\n"
        "User: 'food'\n→ 'a gourmet cheeseburger with melted cheddar and lettuce on a wooden table'\n\n"
        f"Now rewrite this:\n\"{user_query.strip()}\""
    )

    # Запрос к локальному Ollama
    payload = {
        "model": "gemma3:4b",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": system_prompt}
        ],
        "stream": False
    }

    response = requests.post("http://localhost:11434/api/chat", json=payload)
    if response.ok:
        data = response.json()
        return data["message"]["content"].strip()
    else:
        raise RuntimeError(f"LLM error {response.status_code}: {response.text}")


if __name__ == "__main__":
    print(refine_query_for_clip("wooden table"))