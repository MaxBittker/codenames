import { Agent } from "@mariozechner/pi-agent-core";
import type { AgentTool } from "@mariozechner/pi-agent-core";
import { Type, streamSimple } from "@mariozechner/pi-ai";
import type { Model } from "@mariozechner/pi-ai";
import type { BoardState, Clue, GuessResult } from "../types.js";
import { formatBoardForGuesser, getUnrevealedWords } from "./shared.js";

const GUESSER_SYSTEM_PROMPT = `You are a Codenames guesser. You are playing cooperatively with a cluegiver to find all RED words on the board.

RULES:
- You can see the board but NOT the secret key grid.
- Your cluegiver has given you a one-word clue and a number.
- The number tells you how many board words the clue relates to.
- You may make up to (number + 1) guesses per turn.
- You should guess words you think are RED based on the clue.
- If you guess correctly (Red), you may continue guessing.
- If you guess wrong (Blue, Civilian, or Assassin), your turn ends.
- You can stop guessing early if you're unsure, to avoid hitting a bad word.

STRATEGY:
- Start with the word you're MOST confident about.
- If the clue number is 2, the cluegiver likely intended 2 specific words.
- After guessing the intended words, you get 1 bonus guess - only use it if very confident.
- Stop guessing if you're not sure - it's better to be safe than hit the Assassin.
- Consider what words the cluegiver would AVOID connecting to (those are likely Blue or Assassin).

Use the make_guess tool to guess a word, or stop_guessing tool to end your turn.`;

export interface GuessingResult {
  guesses: Array<{ word: string; result: GuessResult }>;
  stoppedEarly: boolean;
}

const guessParams = Type.Object({
  word: Type.String({ description: "The board word you want to guess (must be an unrevealed word on the board)" }),
});

const stopParams = Type.Object({});

export function createGuesserAgent(model: Model<any>, apiKey?: string) {
  let guessesThisTurn: Array<{ word: string; result: GuessResult }> = [];
  let maxGuesses = 0;
  let turnOver = false;
  let stoppedEarly = false;
  let evaluateGuess: ((word: string) => GuessResult) | null = null;
  let currentBoard: BoardState | null = null;

  function endTurn() {
    turnOver = true;
    // Abort the agent loop so it doesn't keep calling tools
    agent.abort();
  }

  const makeGuess: AgentTool<typeof guessParams> = {
    name: "make_guess",
    label: "Make Guess",
    description: "Guess a word on the board that you think is RED.",
    parameters: guessParams,
    execute: async (_toolCallId, params) => {
      if (turnOver) {
        return {
          content: [{ type: "text" as const, text: "Your turn is already over. No more guesses allowed." }],
          details: {},
        };
      }

      const word = params.word.toUpperCase().trim();
      const unrevealed = getUnrevealedWords(currentBoard!);
      const unrevealedUpper = unrevealed.map(w => w.toUpperCase());

      if (!unrevealedUpper.includes(word)) {
        throw new Error(
          `"${word}" is not a valid unrevealed word on the board. Valid words: ${unrevealed.join(", ")}`
        );
      }

      const result = evaluateGuess!(word);
      guessesThisTurn.push({ word, result });

      if (result.type === "correct") {
        const remaining = maxGuesses - guessesThisTurn.length;
        if (remaining <= 0) {
          endTurn();
          return {
            content: [{ type: "text" as const, text: `"${word}" is RED! Correct! You've used all your guesses. Turn over.` }],
            details: { result },
          };
        }
        return {
          content: [{ type: "text" as const, text: `"${word}" is RED! Correct! ${remaining} guess(es) remaining. You may guess again or stop.` }],
          details: { result },
        };
      }

      endTurn();
      if (result.type === "assassin") {
        return {
          content: [{ type: "text" as const, text: `"${word}" is the ASSASSIN! Game over!` }],
          details: { result },
        };
      }
      if (result.type === "wrong") {
        return {
          content: [{ type: "text" as const, text: `"${word}" is ${result.color}. Wrong guess. Turn over.` }],
          details: { result },
        };
      }
      // invalid (shouldn't reach here due to validation above, but satisfies TS)
      return {
        content: [{ type: "text" as const, text: `"${word}" is not valid: ${result.reason}. Turn over.` }],
        details: { result },
      };
    },
  };

  const stopGuessing: AgentTool<typeof stopParams> = {
    name: "stop_guessing",
    label: "Stop Guessing",
    description: "End your turn without making another guess. Use this if you're unsure about remaining guesses.",
    parameters: stopParams,
    execute: async () => {
      stoppedEarly = true;
      endTurn();
      return {
        content: [{ type: "text" as const, text: "You chose to stop guessing. Turn over." }],
        details: {},
      };
    },
  };

  const agent = new Agent({
    initialState: {
      systemPrompt: GUESSER_SYSTEM_PROMPT,
      model,
      tools: [makeGuess, stopGuessing],
    },
    streamFn: streamSimple,
    getApiKey: apiKey ? () => apiKey : undefined,
  });

  async function doGuessing(
    board: BoardState,
    clue: Clue,
    turnNumber: number,
    evaluate: (word: string) => GuessResult,
  ): Promise<GuessingResult> {
    guessesThisTurn = [];
    maxGuesses = clue.number + 1;
    turnOver = false;
    stoppedEarly = false;
    evaluateGuess = evaluate;
    currentBoard = board;

    agent.clearMessages();

    const boardView = formatBoardForGuesser(board);
    const prompt = `Turn ${turnNumber}. The cluegiver gave the clue: "${clue.word}" for ${clue.number}.

You may make up to ${maxGuesses} guesses this turn.

${boardView}

Make your guesses using the make_guess tool, or use stop_guessing to end your turn.`;

    await agent.prompt(prompt);

    return { guesses: guessesThisTurn, stoppedEarly };
  }

  return { agent, doGuessing };
}
