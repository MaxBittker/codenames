"""Evaluate OpenToM — narrative-based Theory of Mind benchmark."""
import asyncio
import re
from openai import AsyncOpenAI
from datasets import load_dataset

async def eval_opentom(model: str, base_url: str, n: int = 100):
    client = AsyncOpenAI(base_url=base_url, api_key="dummy")
    ds = load_dataset("SeacowX/OpenToM", split="Long")
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    
    correct = 0
    total = 0
    by_type = {}
    sem = asyncio.Semaphore(16)
    
    async def eval_one(ex):
        nonlocal correct, total
        q = ex["question"]
        question_text = q["question"]
        answer = q["answer"]
        qtype = q["type"]
        
        prompt = f"""Read the story and answer the question.

Story:
{ex['narrative']}

Question: {question_text}

Choose the best answer from: very positive, positive, neutral, negative, very negative
OR if the question asks about a location, give the location name.

Answer with ONLY the answer. Nothing else."""

        async with sem:
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=64,
                    temperature=0.0,
                )
                pred = resp.choices[0].message.content.strip().lower()
                ans = answer.lower().strip()
                
                # Fuzzy match
                is_correct = ans in pred or pred in ans
                
                if qtype not in by_type:
                    by_type[qtype] = {"correct": 0, "total": 0}
                by_type[qtype]["total"] += 1
                if is_correct:
                    correct += 1
                    by_type[qtype]["correct"] += 1
                total += 1
            except Exception as e:
                total += 1
    
    tasks = [eval_one(ex) for ex in ds]
    await asyncio.gather(*tasks)
    
    acc = correct / total if total > 0 else 0
    print(f"OpenToM Accuracy: {correct}/{total} = {acc:.3f}")
    for qtype, counts in sorted(by_type.items()):
        qacc = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
        print(f"  {qtype}: {counts['correct']}/{counts['total']} = {qacc:.3f}")
    return acc

if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-4B-Instruct-2507"
    base_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000/v1"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    asyncio.run(eval_opentom(model, base_url, n))
