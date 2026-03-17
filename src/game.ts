import type { BoardState, CardColor, Clue, GameEvent, GameOptions, GameOutcome, GuessResult } from "./types.js";
import { createBoard } from "./board.js";
import { createCluegiverAgent } from "./agents/cluegiver.js";
import { createGuesserAgent } from "./agents/guesser.js";

export function evaluateGuess(board: BoardState, word: string): GuessResult {
  const upperWord = word.toUpperCase();
  const index = board.words.findIndex(w => w.toUpperCase() === upperWord);

  if (index === -1) {
    return { type: "invalid", word, reason: "Word not found on board" };
  }
  if (board.revealed[index] !== null) {
    return { type: "invalid", word, reason: "Word already revealed" };
  }

  const color = board.keyGrid[index];
  board.revealed[index] = color;

  if (color === "Red") {
    return { type: "correct", word: board.words[index], color: "Red" };
  }
  if (color === "Assassin") {
    return { type: "assassin", word: board.words[index] };
  }
  return { type: "wrong", word: board.words[index], color };
}

export function countRemaining(board: BoardState, color: CardColor): number {
  let count = 0;
  for (let i = 0; i < 25; i++) {
    if (board.revealed[i] === null && board.keyGrid[i] === color) {
      count++;
    }
  }
  return count;
}

export async function runGame(options: GameOptions): Promise<GameOutcome> {
  const { model, onEvent, apiKey, maxTurns = 25 } = options;

  const board = createBoard(options.words, options.keyGrid);
  const emit = (event: GameEvent) => onEvent?.(event);

  emit({ type: "game_start", words: [...board.words] });

  const cluegiver = createCluegiverAgent(model, apiKey);
  const guesser = createGuesserAgent(model, apiKey);

  let turnsUsed = 0;

  for (let turn = 1; turn <= maxTurns; turn++) {
    const redRemaining = countRemaining(board, "Red");
    if (redRemaining === 0) {
      const outcome: GameOutcome = { result: "win", turnsUsed, redFound: 8 };
      emit({ type: "game_end", outcome });
      return outcome;
    }

    const blueRemaining = countRemaining(board, "Blue");
    emit({ type: "turn_start", turnNumber: turn, redRemaining, blueRemaining });
    turnsUsed = turn;

    let clue: Clue;
    try {
      const t0 = Date.now();
      clue = await cluegiver.getClue(board, turn);
      emit({ type: "clue_given", clue, elapsedMs: Date.now() - t0 });
    } catch (err) {
      const redFound = 8 - countRemaining(board, "Red");
      const outcome: GameOutcome = {
        result: "error",
        turnsUsed,
        redFound,
        message: `Cluegiver error: ${err instanceof Error ? err.message : String(err)}`,
      };
      emit({ type: "game_end", outcome });
      return outcome;
    }

    let assassinHit = false;
    const evaluate = (word: string): GuessResult => {
      const result = evaluateGuess(board, word);
      emit({ type: "guess_made", guess: word, result });
      if (result.type === "assassin") {
        assassinHit = true;
      }
      return result;
    };

    try {
      const t1 = Date.now();
      const guessingResult = await guesser.doGuessing(board, clue, turn, evaluate);
      emit({ type: "guessing_done", elapsedMs: Date.now() - t1 });
      if (guessingResult.stoppedEarly) {
        emit({ type: "guess_stopped" });
      }
    } catch (err) {
      const redFound = 8 - countRemaining(board, "Red");
      const outcome: GameOutcome = {
        result: "error",
        turnsUsed,
        redFound,
        message: `Guesser error: ${err instanceof Error ? err.message : String(err)}`,
      };
      emit({ type: "game_end", outcome });
      return outcome;
    }

    emit({ type: "turn_end", turnNumber: turn });

    if (assassinHit) {
      const redFound = 8 - countRemaining(board, "Red");
      const assassinIndex = board.keyGrid.indexOf("Assassin");
      const outcome: GameOutcome = {
        result: "loss_assassin",
        turnsUsed,
        redFound,
        assassinWord: board.words[assassinIndex],
      };
      emit({ type: "game_end", outcome });
      return outcome;
    }

    if (countRemaining(board, "Red") === 0) {
      const outcome: GameOutcome = { result: "win", turnsUsed, redFound: 8 };
      emit({ type: "game_end", outcome });
      return outcome;
    }

    if (countRemaining(board, "Blue") === 0) {
      const redFound = 8 - countRemaining(board, "Red");
      const outcome: GameOutcome = { result: "loss_blue", turnsUsed, redFound };
      emit({ type: "game_end", outcome });
      return outcome;
    }
  }

  const redFound = 8 - countRemaining(board, "Red");
  const outcome: GameOutcome = {
    result: "error",
    turnsUsed,
    redFound,
    message: `Max turns (${maxTurns}) exceeded`,
  };
  emit({ type: "game_end", outcome });
  return outcome;
}
