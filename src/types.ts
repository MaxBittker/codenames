import type { Model } from "@mariozechner/pi-ai";

export type CardColor = "Red" | "Blue" | "Civilian" | "Assassin";

export interface BoardState {
  words: string[];
  keyGrid: CardColor[];
  revealed: (CardColor | null)[];
}

export interface Clue {
  word: string;
  number: number;
}

export type GuessResult =
  | { type: "correct"; word: string; color: "Red" }
  | { type: "wrong"; word: string; color: "Blue" | "Civilian" }
  | { type: "assassin"; word: string }
  | { type: "invalid"; word: string; reason: string };

export type GameOutcome =
  | { result: "win"; turnsUsed: number; redFound: number }
  | { result: "loss_assassin"; turnsUsed: number; redFound: number; assassinWord: string }
  | { result: "loss_blue"; turnsUsed: number; redFound: number }
  | { result: "error"; turnsUsed: number; redFound: number; message: string };

export type GameEvent =
  | { type: "game_start"; words: string[] }
  | { type: "turn_start"; turnNumber: number; redRemaining: number; blueRemaining: number }
  | { type: "clue_given"; clue: Clue; elapsedMs: number }
  | { type: "guess_made"; guess: string; result: GuessResult }
  | { type: "guess_stopped" }
  | { type: "guessing_done"; elapsedMs: number }
  | { type: "turn_end"; turnNumber: number }
  | { type: "game_end"; outcome: GameOutcome };

export interface GameOptions {
  model: Model<any>;
  words?: string[];
  keyGrid?: CardColor[];
  onEvent?: (event: GameEvent) => void;
  apiKey?: string;
  maxTurns?: number;
}
