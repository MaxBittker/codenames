"""Interactive Codenames UI — play as guesser or cluegiver with a deployed model.

Usage:
    python play.py                          # uses base model
    python play.py --adapter ADAPTER_ID     # uses trained adapter
    python play.py --role guesser           # you guess, model gives clues (default)
    python play.py --role cluegiver         # you give clues, model guesses
    python play.py --board-size 10          # custom board size
"""

import argparse
import json
import os
import re
import sys
from random import Random

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Add codenames package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "environments", "codenames"))
from codenames.game import create_board, evaluate_guess, count_remaining, format_board_for_cluegiver, format_board_for_guesser
from codenames.types import BoardConfig, BoardSamplingConfig, BoardState
from codenames.codenames import CLUEGIVER_SYSTEM_PROMPT, GUESSER_SYSTEM_PROMPT, parse_guesses, _parse_clue_block, _validate_clue

import uvicorn

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE = "https://api.pinference.ai/api/v1"
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

parser_cli = argparse.ArgumentParser()
parser_cli.add_argument("--adapter", default=None, help="Adapter ID for deployed model")
parser_cli.add_argument("--role", default="guesser", choices=["guesser", "cluegiver"], help="Your role")
parser_cli.add_argument("--board-size", type=int, default=10, help="Board size")
parser_cli.add_argument("--num-red", type=int, default=5, help="Number of red words")
parser_cli.add_argument("--port", type=int, default=8899, help="Server port")

app = FastAPI()

# Global game state
GAME = {}
ARGS = None


def get_api_key():
    key = os.environ.get("PRIME_API_KEY", "")
    if not key:
        # Read from prime CLI config
        config_path = os.path.expanduser("~/.prime/config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
                key = config.get("api_key", "")
    return key


def get_model_id():
    if ARGS and ARGS.adapter:
        return f"{BASE_MODEL}:{ARGS.adapter}"
    return BASE_MODEL


async def call_model(messages: list[dict], max_tokens: int = 2048) -> str:
    api_key = get_api_key()
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{API_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": get_model_id(),
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.7,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def new_game():
    board_size = ARGS.board_size if ARGS else 10
    num_red = ARGS.num_red if ARGS else 5
    num_blue = board_size - num_red - 1
    config = BoardConfig(board_size=board_size, num_red=num_red, num_blue=num_blue, num_assassin=1)
    board = create_board(config=config)
    GAME.clear()
    GAME.update({
        "board": board.to_dict(),
        "round": 0,
        "history": [],
        "game_over": False,
        "outcome": None,
        "last_clue": None,
        "role": ARGS.role if ARGS else "guesser",
    })
    return get_game_state()


def get_game_state():
    board = BoardState.from_dict(GAME["board"])
    role = GAME["role"]
    words = []
    for i, (word, color, revealed) in enumerate(zip(board.words, board.key_grid, board.revealed)):
        w = {"word": word, "revealed": revealed is not None, "color": color if revealed else None}
        # Cluegiver always sees colors; guesser sees colors only after reveal
        if role == "cluegiver":
            w["true_color"] = color
        elif revealed:
            w["true_color"] = color
        words.append(w)
    return {
        "words": words,
        "round": GAME["round"],
        "history": GAME["history"],
        "game_over": GAME["game_over"],
        "outcome": GAME["outcome"],
        "role": role,
        "last_clue": GAME["last_clue"],
        "num_red": sum(1 for c in board.key_grid if c == "Red"),
        "red_found": sum(1 for c, r in zip(board.key_grid, board.revealed) if c == "Red" and r is not None),
    }


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    return GAME_HTML


@app.post("/api/new-game")
async def api_new_game():
    return new_game()


@app.get("/api/state")
async def api_state():
    return get_game_state()


class ClueRequest(BaseModel):
    word: str
    number: int


class GuessRequest(BaseModel):
    guesses: list[str]


@app.post("/api/model-clue")
async def api_model_clue():
    """Model gives a clue (player is guesser)."""
    board = BoardState.from_dict(GAME["board"])
    board_text = format_board_for_cluegiver(board)
    num_red = sum(1 for c in board.key_grid if c == "Red")

    parts = []
    if GAME["history"]:
        parts.append(f"Round {GAME['round'] + 1}. Red found so far: {num_red - count_remaining(board, 'Red')}/{num_red}.")
        parts.append("")
        parts.append("Previous rounds:")
        for rnd in GAME["history"]:
            parts.append(f"  Clue: \"{rnd['clue']['word']}\" for {rnd['clue']['number']}")
            for r in rnd["results"]:
                parts.append(f"    {r}")
        parts.append("")
    parts.append(f"Current board:\n{board_text}")

    messages = [
        {"role": "system", "content": CLUEGIVER_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(parts)},
    ]

    response = await call_model(messages)

    # Parse the clue
    clue_match = re.search(r"<clue>(.*?)</clue>", response, re.DOTALL)
    if not clue_match:
        return {"error": "Model failed to produce a valid clue", "raw": response}

    try:
        clue_word, clue_number, target_words = _parse_clue_block(clue_match.group(1))
    except ValueError as e:
        return {"error": str(e), "raw": response}

    GAME["round"] += 1
    GAME["last_clue"] = {"word": clue_word, "number": clue_number, "targets": [w.upper() for w in target_words]}

    return {
        "clue_word": clue_word,
        "clue_number": clue_number,
        "targets": target_words,
        "reasoning": response,
    }


@app.post("/api/player-clue")
async def api_player_clue(req: ClueRequest):
    """Player gives a clue (player is cluegiver)."""
    GAME["round"] += 1
    GAME["last_clue"] = {"word": req.word.upper(), "number": req.number, "targets": []}
    return {"ok": True}


@app.post("/api/model-guess")
async def api_model_guess():
    """Model guesses (player is cluegiver)."""
    board = BoardState.from_dict(GAME["board"])
    clue = GAME["last_clue"]
    num_red = sum(1 for c in board.key_grid if c == "Red")
    max_guesses = clue["number"] + 1

    parts = []
    if GAME["history"]:
        parts.append(f"Round {GAME['round']}. Red found so far: {num_red - count_remaining(board, 'Red')}/{num_red}.")
        parts.append("")
        parts.append("Previous rounds:")
        for rnd in GAME["history"]:
            parts.append(f"  Clue: \"{rnd['clue']['word']}\" for {rnd['clue']['number']}")
            for r in rnd["results"]:
                parts.append(f"    {r}")
        parts.append("")
    parts.append(f'Clue: "{clue["word"]}" for {clue["number"]}')
    parts.append(f"You may guess up to {max_guesses} words.")
    parts.append("")
    parts.append(format_board_for_guesser(board))

    messages = [
        {"role": "system", "content": GUESSER_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(parts)},
    ]

    response = await call_model(messages)
    unrevealed = [w for w, r in zip(board.words, board.revealed) if r is None]
    guesses = parse_guesses(response, unrevealed, max_guesses)

    # Process guesses
    results = []
    for word, reason in guesses:
        gr = evaluate_guess(board, word)
        results.append({"word": gr.word, "type": gr.type, "color": gr.color, "reason": reason})
        if gr.type != "correct":
            break

    GAME["board"] = board.to_dict()
    round_results = [f"{r['word']} ({r['color'] or r['type']})" for r in results]
    GAME["history"].append({"clue": clue, "results": round_results})
    GAME["last_clue"] = None

    # Check terminal
    if any(r["type"] == "assassin" for r in results):
        GAME["game_over"] = True
        GAME["outcome"] = "LOSS — Assassin hit!"
    elif count_remaining(board, "Red") == 0:
        GAME["game_over"] = True
        GAME["outcome"] = f"WIN — All reds found in {GAME['round']} rounds!"

    return {"guesses": results, "reasoning": response, **get_game_state()}


@app.post("/api/player-guess")
async def api_player_guess(req: GuessRequest):
    """Player guesses (player is guesser)."""
    board = BoardState.from_dict(GAME["board"])
    clue = GAME["last_clue"]
    results = []

    for word in req.guesses:
        gr = evaluate_guess(board, word)
        results.append({"word": gr.word, "type": gr.type, "color": gr.color})
        if gr.type != "correct":
            break

    GAME["board"] = board.to_dict()
    round_results = [f"{r['word']} ({r['color'] or r['type']})" for r in results]
    GAME["history"].append({"clue": clue, "results": round_results})
    GAME["last_clue"] = None

    if any(r["type"] == "assassin" for r in results):
        GAME["game_over"] = True
        GAME["outcome"] = "LOSS — Assassin hit!"
    elif count_remaining(board, "Red") == 0:
        GAME["game_over"] = True
        GAME["outcome"] = f"WIN — All reds found in {GAME['round']} rounds!"

    return {"guesses": results, **get_game_state()}


# ---------------------------------------------------------------------------
# HTML UI
# ---------------------------------------------------------------------------

GAME_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Codenames — Play with AI</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, -apple-system, sans-serif; background: #1a1a2e; color: #eee; min-height: 100vh; }
.container { max-width: 900px; margin: 0 auto; padding: 20px; }
h1 { text-align: center; margin-bottom: 8px; font-size: 1.6em; }
.subtitle { text-align: center; color: #888; margin-bottom: 20px; font-size: 0.9em; }
.board { display: grid; gap: 8px; margin: 20px 0; }
.card {
    padding: 14px 8px; text-align: center; border-radius: 8px;
    font-weight: 700; font-size: 0.85em; cursor: default;
    background: #2a2a4a; color: #ccc; border: 2px solid #3a3a5a;
    transition: all 0.2s; text-transform: uppercase; letter-spacing: 0.5px;
}
.card.selectable { cursor: pointer; }
.card.selectable:hover { border-color: #7c7cf0; transform: translateY(-2px); box-shadow: 0 4px 12px rgba(124,124,240,0.3); }
.card.selected { border-color: #f0c040; box-shadow: 0 0 12px rgba(240,192,64,0.4); }
.card.revealed-Red { background: #c0392b; color: white; border-color: #e74c3c; }
.card.revealed-Blue { background: #2980b9; color: white; border-color: #3498db; }
.card.revealed-Assassin { background: #1a1a1a; color: #e74c3c; border-color: #c0392b; }
/* Cluegiver sees colors as subtle tints */
.card.hint-Red { background: #3d2020; border-color: #6a3030; }
.card.hint-Blue { background: #1a2a3d; border-color: #2a4a6a; }
.card.hint-Assassin { background: #2a1a1a; border-color: #5a2020; }
.controls { display: flex; gap: 12px; align-items: center; justify-content: center; flex-wrap: wrap; margin: 16px 0; }
button {
    padding: 10px 20px; border: none; border-radius: 6px; font-size: 0.9em;
    cursor: pointer; font-weight: 600; transition: all 0.15s;
}
.btn-primary { background: #6c5ce7; color: white; }
.btn-primary:hover { background: #7c6cf7; }
.btn-primary:disabled { background: #444; cursor: not-allowed; }
.btn-danger { background: #d63031; color: white; }
.btn-danger:hover { background: #e64040; }
.btn-secondary { background: #444; color: #ccc; }
.btn-secondary:hover { background: #555; }
input[type=text], input[type=number] {
    padding: 10px 14px; border: 2px solid #3a3a5a; border-radius: 6px;
    background: #2a2a4a; color: #eee; font-size: 0.9em;
}
input[type=number] { width: 60px; }
.status { text-align: center; padding: 12px; border-radius: 8px; margin: 12px 0; font-weight: 600; }
.status.win { background: #27ae60; }
.status.loss { background: #c0392b; }
.status.info { background: #2a2a4a; border: 1px solid #3a3a5a; }
.clue-display { text-align: center; font-size: 1.3em; margin: 16px 0; }
.clue-display .word { color: #f0c040; font-weight: 700; font-size: 1.4em; }
.clue-display .number { color: #7c7cf0; }
.history { margin: 20px 0; }
.history h3 { margin-bottom: 8px; color: #888; }
.round-entry { padding: 8px 12px; background: #2a2a4a; border-radius: 6px; margin: 4px 0; font-size: 0.85em; }
.thinking { text-align: center; color: #888; padding: 20px; }
.thinking::after { content: '...'; animation: dots 1.5s infinite; }
@keyframes dots { 0%{content:'.'} 33%{content:'..'} 66%{content:'...'} }
.reasoning-toggle { font-size: 0.8em; color: #666; cursor: pointer; margin-top: 4px; }
.reasoning-box { background: #1a1a2e; border: 1px solid #3a3a5a; border-radius: 6px; padding: 12px; margin: 8px 0; font-size: 0.8em; white-space: pre-wrap; max-height: 200px; overflow-y: auto; display: none; }
.score { text-align: center; color: #888; font-size: 0.9em; }
</style>
</head>
<body>
<div class="container">
    <h1>Codenames</h1>
    <div class="subtitle" id="roleLabel"></div>
    <div class="score" id="score"></div>
    <div id="statusBar"></div>
    <div id="clueArea"></div>
    <div class="board" id="board"></div>
    <div class="controls" id="controls"></div>
    <div class="history" id="historyArea"></div>
</div>
<script>
let state = null;
let selectedWords = [];
let loading = false;

async function api(path, body) {
    const opts = body ? {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)} : {method: path.includes('state') ? 'GET' : 'POST'};
    const r = await fetch('/api/' + path, opts);
    return r.json();
}

async function newGame() {
    state = await api('new-game', {});
    selectedWords = [];
    render();
    if (state.role === 'guesser') await getModelClue();
}

async function getModelClue() {
    loading = true; render();
    const r = await api('model-clue', {});
    loading = false;
    if (r.error) { alert('Model error: ' + r.error); return; }
    state = await api('state');
    state._reasoning = r.reasoning;
    render();
}

async function submitPlayerGuesses() {
    if (!selectedWords.length) return;
    loading = true; render();
    const r = await api('player-guess', {guesses: selectedWords});
    loading = false;
    state = r;
    selectedWords = [];
    render();
    if (!state.game_over && state.role === 'guesser') {
        setTimeout(() => getModelClue(), 500);
    }
}

async function submitPlayerClue() {
    const word = document.getElementById('clueWord').value.trim();
    const num = parseInt(document.getElementById('clueNum').value);
    if (!word || !num) return;
    await api('player-clue', {word, number: num});
    state = await api('state');
    loading = true; render();
    const r = await api('model-guess', {});
    loading = false;
    state = r;
    state._reasoning = r.reasoning;
    render();
    if (!state.game_over && state.role === 'cluegiver') render();
}

function toggleWord(word) {
    if (state.game_over || loading) return;
    const idx = selectedWords.indexOf(word);
    if (idx >= 0) selectedWords.splice(idx, 1);
    else selectedWords.push(word);
    render();
}

function render() {
    if (!state) return;
    const {words, round, history, game_over, outcome, role, last_clue, num_red, red_found} = state;

    document.getElementById('roleLabel').textContent = role === 'guesser'
        ? 'You are the GUESSER — the AI gives clues'
        : 'You are the CLUEGIVER — the AI guesses';

    document.getElementById('score').textContent = `Round ${round} — ${red_found}/${num_red} reds found`;

    // Board
    const cols = Math.ceil(Math.sqrt(words.length));
    const boardEl = document.getElementById('board');
    boardEl.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
    boardEl.innerHTML = words.map(w => {
        let cls = 'card';
        if (w.revealed) cls += ` revealed-${w.color}`;
        else if (w.true_color && !w.revealed) cls += ` hint-${w.true_color}`;
        if (role === 'guesser' && !w.revealed && !game_over && last_clue) cls += ' selectable';
        if (selectedWords.includes(w.word)) cls += ' selected';
        const onclick = (role === 'guesser' && !w.revealed && !game_over && last_clue)
            ? `onclick="toggleWord('${w.word}')"`
            : '';
        return `<div class="${cls}" ${onclick}>${w.word}</div>`;
    }).join('');

    // Status
    const statusEl = document.getElementById('statusBar');
    if (game_over) {
        const cls = outcome.startsWith('WIN') ? 'win' : 'loss';
        statusEl.innerHTML = `<div class="status ${cls}">${outcome}</div>`;
    } else if (loading) {
        statusEl.innerHTML = `<div class="thinking">AI is thinking</div>`;
    } else {
        statusEl.innerHTML = '';
    }

    // Clue area
    const clueEl = document.getElementById('clueArea');
    if (last_clue && role === 'guesser') {
        clueEl.innerHTML = `<div class="clue-display">
            Clue: <span class="word">${last_clue.word}</span> for <span class="number">${last_clue.number}</span>
        </div>`;
        if (state._reasoning) {
            const id = 'reasoning-' + round;
            clueEl.innerHTML += `<div class="reasoning-toggle" onclick="document.getElementById('${id}').style.display=document.getElementById('${id}').style.display==='none'?'block':'none'">Show AI reasoning</div>
            <div class="reasoning-box" id="${id}">${state._reasoning.replace(/</g,'&lt;').replace(/>/g,'&gt;')}</div>`;
        }
    } else {
        clueEl.innerHTML = '';
    }

    // Controls
    const ctrlEl = document.getElementById('controls');
    if (game_over) {
        ctrlEl.innerHTML = `<button class="btn-primary" onclick="newGame()">New Game</button>`;
    } else if (role === 'guesser' && last_clue && !loading) {
        ctrlEl.innerHTML = `
            <button class="btn-primary" onclick="submitPlayerGuesses()" ${selectedWords.length ? '' : 'disabled'}>
                Submit ${selectedWords.length} guess${selectedWords.length !== 1 ? 'es' : ''}
            </button>`;
    } else if (role === 'cluegiver' && !last_clue && !loading) {
        ctrlEl.innerHTML = `
            <input type="text" id="clueWord" placeholder="Clue word">
            <input type="number" id="clueNum" value="2" min="1" max="8">
            <button class="btn-primary" onclick="submitPlayerClue()">Give Clue</button>`;
    } else if (!loading) {
        ctrlEl.innerHTML = '';
    }

    // History
    const histEl = document.getElementById('historyArea');
    if (history.length) {
        histEl.innerHTML = '<h3>Round History</h3>' + history.map((h, i) =>
            `<div class="round-entry"><strong>Round ${i+1}:</strong> "${h.clue.word}" for ${h.clue.number} → ${h.results.join(', ')}</div>`
        ).join('');
    } else {
        histEl.innerHTML = '';
    }
}

// Start
newGame();
</script>
</body>
</html>"""


if __name__ == "__main__":
    ARGS = parser_cli.parse_args()
    new_game()
    print(f"\n  Codenames — playing as {ARGS.role}")
    print(f"  Model: {get_model_id()}")
    print(f"  Board: {ARGS.board_size} words, {ARGS.num_red} red")
    print(f"\n  Open http://localhost:{ARGS.port}\n")
    uvicorn.run(app, host="0.0.0.0", port=ARGS.port, log_level="warning")
