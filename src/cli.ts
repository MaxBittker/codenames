import "dotenv/config";
import { runGame } from "./game.js";
import { createModelFromEnv } from "./model.js";
import type { GameEvent } from "./types.js";

const model = createModelFromEnv();
const apiKey = process.env.LLM_API_KEY;

function onEvent(event: GameEvent): void {
  switch (event.type) {
    case "game_start":
      console.log("=== CODENAMES SOLITAIRE ===\n");
      console.log("Board words:");
      for (let row = 0; row < 5; row++) {
        const words = event.words.slice(row * 5, row * 5 + 5);
        console.log("  " + words.map(w => w.padEnd(18)).join(""));
      }
      console.log();
      break;

    case "turn_start":
      console.log(`--- Turn ${event.turnNumber} (Red: ${event.redRemaining} remaining, Blue: ${event.blueRemaining} remaining) ---`);
      break;

    case "clue_given":
      console.log(`Cluegiver: "${event.clue.word}" for ${event.clue.number} (${(event.elapsedMs / 1000).toFixed(1)}s)`);
      break;

    case "guessing_done":
      console.log(`  Guessing done (${(event.elapsedMs / 1000).toFixed(1)}s)`);
      break;

    case "guess_made": {
      const r = event.result;
      if (r.type === "correct") {
        console.log(`  Guesser guessed "${event.guess}" -> RED (correct!)`);
      } else if (r.type === "assassin") {
        console.log(`  Guesser guessed "${event.guess}" -> ASSASSIN!`);
      } else if (r.type === "wrong") {
        console.log(`  Guesser guessed "${event.guess}" -> ${r.color}`);
      } else {
        console.log(`  Guesser guessed "${event.guess}" -> INVALID: ${r.reason}`);
      }
      break;
    }

    case "guess_stopped":
      console.log("  Guesser chose to stop guessing.");
      break;

    case "turn_end":
      console.log();
      break;

    case "game_end": {
      const o = event.outcome;
      console.log("=== GAME OVER ===");
      if (o.result === "win") {
        console.log(`WIN! Found all 8 Red words in ${o.turnsUsed} turns.`);
      } else if (o.result === "loss_assassin") {
        console.log(`LOSS - Hit the assassin word "${o.assassinWord}". Found ${o.redFound}/8 Red words in ${o.turnsUsed} turns.`);
      } else if (o.result === "loss_blue") {
        console.log(`LOSS - All Blue words revealed. Found ${o.redFound}/8 Red words in ${o.turnsUsed} turns.`);
      } else {
        console.log(`ERROR - ${o.message}. Found ${o.redFound}/8 Red words in ${o.turnsUsed} turns.`);
      }
      break;
    }
  }
}

async function main() {
  const outcome = await runGame({ model, onEvent, apiKey });
  process.exit(outcome.result === "win" ? 0 : 1);
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
