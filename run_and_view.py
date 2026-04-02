"""Run a single multi-turn Codenames game and open an HTML viewer showing both agent trajectories side by side."""

import asyncio
import html
import json
import os
import tempfile
import webbrowser

from openai import AsyncOpenAI


def escape(text: str) -> str:
    return html.escape(text)


def render_html(board_info: dict, rounds: list[dict], outcome: dict) -> str:
    board_cards = ""
    for w, c in zip(board_info["words"], board_info["key_grid"]):
        board_cards += f'<span class="card {c.lower()}">{escape(w)}</span> '

    # Build sequence of steps — alternating columns
    steps = []
    for i, rnd in enumerate(rounds, 1):
        # Cluegiver: prompt + response (left column active)
        steps.append(f"""
        <div class="step">
            <div class="lane left active">
                <div class="agent-tag cluegiver-tag">Cluegiver — Round {i}</div>
                <div class="block prompt-block">
                    <div class="block-label">sees</div>
                    <pre>{escape(rnd["cluegiver_prompt_user"])}</pre>
                </div>
                <div class="block response-block cluegiver-response">
                    <div class="block-label">responds</div>
                    <pre>{escape(rnd["cluegiver_response"])}</pre>
                </div>
            </div>
            <div class="lane right dimmed"></div>
        </div>
        <div class="arrow-row"><div class="arrow">&#x25BC; passes clue</div></div>
        """)

        # Guesser: prompt + response + results (right column active)
        if rnd.get("guesser_prompt_user"):
            results_html = ""
            for r in rnd["results"]:
                rtype = r["type"]
                cls = "result-correct" if rtype == "correct" else "result-assassin" if rtype == "assassin" else "result-wrong"
                label = f'{r["word"]} → {r.get("color", rtype).upper()}'
                results_html += f'<span class="{cls}">{escape(label)}</span> '

            steps.append(f"""
            <div class="step">
                <div class="lane left dimmed"></div>
                <div class="lane right active">
                    <div class="agent-tag guesser-tag">Guesser — Round {i}</div>
                    <div class="block prompt-block">
                        <div class="block-label">sees</div>
                        <pre>{escape(rnd["guesser_prompt_user"])}</pre>
                    </div>
                    <div class="block response-block guesser-response">
                        <div class="block-label">responds</div>
                        <pre>{escape(rnd["guesser_response"])}</pre>
                    </div>
                    <div class="results">{results_html}</div>
                </div>
            </div>
            """)
        else:
            steps.append(f"""
            <div class="step">
                <div class="lane left dimmed"></div>
                <div class="lane right active">
                    <div class="agent-tag guesser-tag">Guesser — Round {i}</div>
                    <div class="block error-block">No guesser turn (cluegiver format failure)</div>
                </div>
            </div>
            """)

        # Arrow back to cluegiver if not last round
        if i < len(rounds):
            steps.append('<div class="arrow-row"><div class="arrow">&#x25BC; next round</div></div>')

    outcome_class = "outcome-win" if outcome["won"] else "outcome-loss"

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Codenames Game Viewer</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace; background: #0d1117; color: #c9d1d9; font-size: 13px; }}

.topbar {{ background: #161b22; padding: 14px 24px; border-bottom: 1px solid #30363d; position: sticky; top: 0; z-index: 10; }}
.topbar h1 {{ font-size: 15px; font-weight: 600; color: #e6edf3; display: inline; }}
.topbar .scores {{ display: inline; margin-left: 24px; font-size: 12px; color: #8b949e; }}
.topbar .scores span {{ color: #e6edf3; font-weight: 600; }}

.board {{ padding: 10px 24px; background: #161b22; border-bottom: 1px solid #30363d; display: flex; flex-wrap: wrap; gap: 5px; position: sticky; top: 48px; z-index: 9; }}
.card {{ padding: 3px 9px; border-radius: 4px; font-size: 11px; font-weight: 600; }}
.card.red {{ background: #3d1f1f; color: #f85149; border: 1px solid #f8514933; }}
.card.blue {{ background: #1f2d3d; color: #58a6ff; border: 1px solid #58a6ff33; }}
.card.assassin {{ background: #2d1f3d; color: #d2a8ff; border: 1px solid #d2a8ff33; }}

.outcome-banner {{ padding: 10px 24px; text-align: center; font-weight: 700; font-size: 14px; }}
.outcome-win {{ background: #1a2e1a; color: #3fb950; }}
.outcome-loss {{ background: #2e1a1a; color: #f85149; }}

.flow {{ max-width: 1100px; margin: 0 auto; padding: 24px 16px 80px 16px; }}

/* Each step is two side-by-side lanes */
.step {{ display: flex; gap: 12px; }}
.lane {{ flex: 1; min-width: 0; padding: 0 4px; transition: opacity 0.2s; }}
.lane.dimmed {{ opacity: 0.15; }}
.lane.active {{ opacity: 1; }}

.agent-tag {{ font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px; padding: 3px 10px; border-radius: 4px; display: inline-block; }}
.cluegiver-tag {{ background: #f8514922; color: #f85149; }}
.guesser-tag {{ background: #58a6ff22; color: #58a6ff; }}

.block {{ margin: 6px 0; border-radius: 6px; padding: 10px 12px; }}
.block-label {{ font-size: 10px; font-weight: 700; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 5px; }}
.prompt-block {{ background: #161b22; border: 1px solid #30363d; }}
.response-block {{ background: #1c2128; border: 1px solid #30363d; }}
.cluegiver-response {{ border-left: 3px solid #f85149; }}
.guesser-response {{ border-left: 3px solid #58a6ff; }}
.error-block {{ background: #2e1a1a; border: 1px solid #f8514933; color: #f85149; padding: 10px 12px; border-radius: 6px; }}

pre {{ white-space: pre-wrap; word-wrap: break-word; font-family: inherit; font-size: 12px; line-height: 1.5; }}

.results {{ margin: 6px 0 4px 0; }}
.results span {{ display: inline-block; padding: 3px 8px; border-radius: 3px; font-size: 11px; font-weight: 600; margin: 2px; }}
.result-correct {{ background: #1a2e1a; color: #3fb950; }}
.result-wrong {{ background: #1f2d3d; color: #58a6ff; }}
.result-assassin {{ background: #2e1a1a; color: #f85149; }}

.arrow-row {{ text-align: center; padding: 8px 0; }}
.arrow {{ display: inline-block; font-size: 11px; color: #484f58; letter-spacing: 0.5px; }}
</style>
</head>
<body>
<div class="topbar">
    <h1>Codenames — {outcome['rounds']} rounds, {outcome['red_found']}/{outcome['num_red']} reds</h1>
    <span class="scores">
        reward: <span>{outcome['game_reward']:.2f}</span>
        &nbsp; efficiency: <span>{outcome['efficiency']:.2f}</span>
        &nbsp; blue_hits: <span>{outcome['blue_hits']}</span>
        &nbsp; avg_clue#: <span>{outcome['avg_clue']:.1f}</span>
    </span>
</div>
<div class="board">{board_cards}</div>

<div class="flow">
    <div class="step" style="margin-bottom:12px;">
        <div class="lane left" style="opacity:1;"><div class="agent-tag cluegiver-tag" style="font-size:12px;">Cluegiver (sees colors)</div></div>
        <div class="lane right" style="opacity:1;"><div class="agent-tag guesser-tag" style="font-size:12px;">Guesser (words only)</div></div>
    </div>
    {''.join(steps)}
</div>

<div class="outcome-banner {outcome_class}">{escape(outcome['summary'])}</div>
</body>
</html>"""


async def main():
    from codenames.codenames import (
        CodenamesEnv,
        _format_guess_result,
        game_reward,
        efficiency_reward,
        win_metric,
        assassin_metric,
        blue_hit_metric,
        rounds_metric,
        avg_clue_number_metric,
        red_found_metric,
    )
    from codenames.game import create_board, count_remaining
    from codenames.types import BoardSamplingConfig, BoardState
    from datasets import Dataset
    from random import Random

    rng = Random(99)
    sampling = BoardSamplingConfig(min_board_size=12, max_board_size=12)
    config = sampling.sample(rng)
    board = create_board(rng=rng, config=config)

    board_info = {
        "words": board.words,
        "key_grid": board.key_grid,
        "num_red": config.num_red,
        "num_blue": config.num_blue,
    }

    dummy_ds = Dataset.from_list(
        [{"prompt": [], "info": {}, "answer": "", "task": "test"}]
    )
    env = CodenamesEnv(
        dataset=dummy_ds,
        eval_dataset=dummy_ds,
        rubric=None,
        max_turns=20,
    )

    state = {
        "info": {
            "words": board.words,
            "key_grid": board.key_grid,
            "board_config": config.to_dict(),
        },
        "extras": {},
    }
    state = await env.setup_state(state)

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model = "gpt-4.1-mini"

    rounds_data = []
    round_num = 0

    while not state.get("game_over", False) and round_num < 10:
        round_num += 1
        rnd = {}

        # Cluegiver
        cluegiver_prompt = await env.build_agent_prompt("cluegiver", state)
        rnd["cluegiver_prompt_user"] = cluegiver_prompt[-1]["content"]

        print(f"Round {round_num}: cluegiver thinking...")
        resp = await client.chat.completions.create(
            model=model, messages=cluegiver_prompt, max_tokens=500,
        )
        cluegiver_text = resp.choices[0].message.content
        rnd["cluegiver_response"] = cluegiver_text

        env._process_cluegiver_turn(cluegiver_text, state)
        if state.get("game_over"):
            rnd["guesser_prompt_user"] = None
            rnd["guesser_response"] = None
            rnd["results"] = []
            rounds_data.append(rnd)
            break

        # Guesser
        guesser_prompt = await env.build_agent_prompt("guesser", state)
        rnd["guesser_prompt_user"] = guesser_prompt[-1]["content"]

        print(f"Round {round_num}: guesser thinking...")
        resp = await client.chat.completions.create(
            model=model, messages=guesser_prompt, max_tokens=500,
        )
        guesser_text = resp.choices[0].message.content
        rnd["guesser_response"] = guesser_text

        env._process_guesser_turn(guesser_text, state)

        # Collect results
        rnd_results = []
        if state.get("round_history"):
            last = state["round_history"][-1]
            for r in last["results"]:
                rnd_results.append({
                    "word": r.word,
                    "type": r.type,
                    "color": r.color or r.type,
                })
        rnd["results"] = rnd_results
        rounds_data.append(rnd)

        current_board = BoardState.from_dict(state["board"])
        red_remaining = count_remaining(current_board, "Red")
        red_found = config.num_red - red_remaining
        print(f"  -> {red_found}/{config.num_red} reds found, {state.get('blue_hit_count', 0)} blue hits")

        if state.get("assassin_hit"):
            print("  -> ASSASSIN HIT")
        elif red_remaining == 0:
            print("  -> ALL REDS FOUND")

    # Score
    won = (await win_metric(state=state)) == 1.0
    current_board = BoardState.from_dict(state["board"])
    outcome = {
        "won": won,
        "summary": "WIN — all reds found!" if won else ("LOSS — assassin hit" if state.get("assassin_hit") else "INCOMPLETE"),
        "rounds": len(state.get("round_history", [])),
        "red_found": state.get("total_red_found", 0),
        "num_red": config.num_red,
        "game_reward": await game_reward(state=state),
        "efficiency": await efficiency_reward(state=state),
        "blue_hits": int(await blue_hit_metric(state=state)),
        "avg_clue": await avg_clue_number_metric(state=state),
    }

    html_content = render_html(board_info, rounds_data, outcome)

    out_path = os.path.join(os.path.dirname(__file__), "game_viewer.html")
    with open(out_path, "w") as f:
        f.write(html_content)
    print(f"\nOpening {out_path}")
    webbrowser.open(f"file://{out_path}")


if __name__ == "__main__":
    asyncio.run(main())
