import type { BoardState } from "../types.js";

export function getUnrevealedWords(board: BoardState): string[] {
  return board.words.filter((_, i) => board.revealed[i] === null);
}

export function formatBoardForCluegiver(board: BoardState): string {
  const red: string[] = [];
  const blue: string[] = [];
  const civilian: string[] = [];
  let assassin = "";

  for (let i = 0; i < 25; i++) {
    if (board.revealed[i] !== null) continue;
    const word = board.words[i];
    switch (board.keyGrid[i]) {
      case "Red": red.push(word); break;
      case "Blue": blue.push(word); break;
      case "Civilian": civilian.push(word); break;
      case "Assassin": assassin = word; break;
    }
  }

  const lines: string[] = [];
  lines.push(`RED words to find (${red.length} remaining): ${red.join(", ")}`);
  lines.push(`BLUE words to AVOID: ${blue.join(", ")}`);
  lines.push(`CIVILIAN words to AVOID: ${civilian.join(", ")}`);
  lines.push(`ASSASSIN word to AVOID: ${assassin}`);

  return lines.join("\n");
}

export function formatBoardForGuesser(board: BoardState): string {
  const unrevealed = getUnrevealedWords(board);
  return `Remaining words: ${unrevealed.join(", ")}`;
}
