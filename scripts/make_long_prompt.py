"""Generate a long-context JSON request file using a real book from PG-19."""

import json
from datasets import load_dataset

ds = load_dataset("deepmind/pg19", split="test", streaming=True)

# Grab the first book that's long enough
for sample in ds:
    text = sample["text"]
    # ~4 chars per token, target 65K+ tokens
    if len(text) >= 260_000:
        # Truncate to ~65K tokens worth
        context = text[:260_000]
        break

prompt = (
    f"Read the following text carefully, then answer the questions below.\n\n"
    f"---\n{context}\n---\n\n"
    f"1. Summarize the main events or themes of this text in 3-5 sentences.\n"
    f"2. Who are the main characters or subjects mentioned?\n"
    f"3. What is the tone or style of the writing?"
)

request = {
    "model": "deepseek-ai/DeepSeek-V3.2-TEE:THINKING",
    "messages": [{"role": "user", "content": prompt}],
    "stream": False,
}

with open("long_prompt_65k.json", "w") as f:
    json.dump(request, f)

print(f"Context length: ~{len(context)//4} tokens ({len(context)} chars)")
print(f"Saved to long_prompt_65k.json")
print(f"Test with: curl -X POST http://localhost:30000/v1/chat/completions -H 'Content-Type: application/json' -d @long_prompt_65k.json")
