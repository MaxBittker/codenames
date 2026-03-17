import "dotenv/config";
import { createRandomMidGameBoard } from "./board.js";
import { createCluegiverAgent } from "./agents/cluegiver.js";
import { createGuesserAgent } from "./agents/guesser.js";
import { evaluateGuess, countRemaining } from "./game.js";
import { createModelFromEnv } from "./model.js";
import { formatBoardForCluegiver } from "./agents/shared.js";
import type { GuessResult } from "./types.js";

const model = createModelFromEnv();
const apiKey = process.env.LLM_API_KEY;

function subscribeAgent(agent: any, label: string) {
  let inThinking = false;

  agent.subscribe((e: any) => {
    if (e.type === "message_update") {
      const ae = e.assistantMessageEvent;
      if (ae.type === "thinking_delta") {
        if (!inThinking) {
          process.stdout.write("\x1b[2m"); // dim
          inThinking = true;
        }
        process.stdout.write(ae.delta);
      } else if (ae.type === "text_delta") {
        if (inThinking) {
          process.stdout.write("\x1b[0m\n"); // reset
          inThinking = false;
        }
        process.stdout.write(ae.delta);
      }
    }
    if (e.type === "message_end") {
      if (inThinking) {
        process.stdout.write("\x1b[0m"); // reset
        inThinking = false;
      }
      console.log();
    }
    if (e.type === "tool_execution_start") {
      if (inThinking) {
        process.stdout.write("\x1b[0m\n");
        inThinking = false;
      }
      console.log(`  [${e.toolName}(${JSON.stringify(e.args)})]`);
    }
  });
}

async function main() {
  const board = createRandomMidGameBoard();

  // Print board state
  console.log("=".repeat(60));
  console.log("BOARD STATE");
  console.log("=".repeat(60));
  console.log(formatBoardForCluegiver(board));
  console.log();

  const redAvailable = countRemaining(board, "Red");
  console.log(`Red available: ${redAvailable}`);
  console.log();

  // Cluegiver
  console.log("=".repeat(60));
  console.log("CLUEGIVER");
  console.log("=".repeat(60));

  const cluegiver = createCluegiverAgent(model, apiKey);
  subscribeAgent(cluegiver.agent, "Cluegiver");

  const t0 = Date.now();
  let clue;
  try {
    clue = await cluegiver.getClue(board, 1);
    console.log(`\n>> Clue: "${clue.word}" for ${clue.number} (${((Date.now() - t0) / 1000).toFixed(1)}s)`);
  } catch (err) {
    console.log(`\n>> Cluegiver failed: ${err instanceof Error ? err.message : err}`);
    process.exit(1);
  }
  console.log();

  // Guesser
  console.log("=".repeat(60));
  console.log("GUESSER");
  console.log("=".repeat(60));

  const guesser = createGuesserAgent(model, apiKey);
  subscribeAgent(guesser.agent, "Guesser");

  const evaluate = (word: string): GuessResult => {
    const result = evaluateGuess(board, word);
    const label = result.type === "correct" ? "RED" :
                  result.type === "assassin" ? "ASSASSIN" :
                  result.type === "wrong" ? result.color :
                  `INVALID: ${result.reason}`;
    console.log(`  >> "${word}" -> ${label}`);
    return result;
  };

  const t1 = Date.now();
  const guessingResult = await guesser.doGuessing(board, clue, 1, evaluate);
  console.log(`\n>> Guessing done (${((Date.now() - t1) / 1000).toFixed(1)}s) | Stopped early: ${guessingResult.stoppedEarly}`);

  // Summary
  const redFound = redAvailable - countRemaining(board, "Red");
  console.log();
  console.log("=".repeat(60));
  console.log(`RESULT: ${redFound}/${redAvailable} red found`);
  console.log("=".repeat(60));
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
