"""Evaluate Hi-ToM (Higher-Order Theory of Mind) benchmark."""
import asyncio
import json
import re
from openai import AsyncOpenAI
from datasets import load_dataset

async def eval_hitom(model: str, base_url: str, n: int = 200):
    client = AsyncOpenAI(base_url=base_url, api_key="dummy")
    ds = load_dataset("Hi-ToM/Hi-ToM_Dataset", split="train")
    
    # Sample n examples
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    
    correct = 0
    total = 0
    
    sem = asyncio.Semaphore(16)
    
    async def eval_one(ex):
        nonlocal correct, total
        prompt = f"""Read the following story and answer the question.

Story:
{ex['story']}

Question: {ex['question']}

Choices:
{ex['choices']}

Answer with ONLY the letter (A, B, C, etc.) of the correct choice. Nothing else."""

        async with sem:
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=32,
                    temperature=0.0,
                )
                answer = resp.choices[0].message.content.strip()
                # Extract letter from response
                letter_match = re.search(r'^([A-O])', answer)
                if letter_match:
                    pred_letter = letter_match.group(1)
                    # Find which choice the answer corresponds to
                    choices = ex['choices'].split(', ')
                    answer_map = {}
                    for c in choices:
                        parts = c.split('. ', 1)
                        if len(parts) == 2:
                            answer_map[parts[1].strip()] = parts[0].strip()
                    
                    correct_letter = answer_map.get(ex['answer'], '')
                    if pred_letter == correct_letter:
                        correct += 1
                total += 1
            except Exception as e:
                total += 1
    
    tasks = [eval_one(ex) for ex in ds]
    await asyncio.gather(*tasks)
    
    acc = correct / total if total > 0 else 0
    print(f"Hi-ToM Accuracy: {correct}/{total} = {acc:.3f}")
    return acc

if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-4B-Instruct-2507"
    base_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000/v1"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    asyncio.run(eval_hitom(model, base_url, n))
