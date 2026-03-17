import type { BoardState, CardColor } from "./types.js";
import { WORD_POOL } from "./wordpool.js";

export function shuffle<T>(arr: readonly T[]): T[] {
  const result = [...arr];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

export function selectWords(pool: readonly string[] = WORD_POOL): string[] {
  return shuffle(pool).slice(0, 25);
}

export function generateKeyGrid(): CardColor[] {
  const colors: CardColor[] = [
    ...Array<CardColor>(8).fill("Red"),
    ...Array<CardColor>(7).fill("Blue"),
    ...Array<CardColor>(9).fill("Civilian"),
    "Assassin",
  ];
  return shuffle(colors);
}

export function createBoard(words?: string[], keyGrid?: CardColor[]): BoardState {
  return {
    words: words ?? selectWords(),
    keyGrid: keyGrid ?? generateKeyGrid(),
    revealed: Array(25).fill(null) as (CardColor | null)[],
  };
}

export function createRandomMidGameBoard(): BoardState {
  const board = createBoard();

  // Pick a random number of cards to pre-reveal (0-12)
  const numToReveal = Math.floor(Math.random() * 13);

  // Gather all card indices
  const indices = Array.from({ length: 25 }, (_, i) => i);
  const shuffled = shuffle(indices);

  let revealed = 0;
  for (const i of shuffled) {
    if (revealed >= numToReveal) break;

    // Never reveal the assassin
    if (board.keyGrid[i] === "Assassin") continue;

    // Skip revealing a red card if it would leave 0 reds unrevealed
    if (board.keyGrid[i] === "Red") {
      const redUnrevealed = board.keyGrid.reduce(
        (count, color, j) => count + (color === "Red" && board.revealed[j] === null ? 1 : 0),
        0,
      );
      if (redUnrevealed <= 1) continue;
    }

    board.revealed[i] = board.keyGrid[i];
    revealed++;
  }

  return board;
}
