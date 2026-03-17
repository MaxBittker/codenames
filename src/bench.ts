import "dotenv/config";
import { createRandomMidGameBoard } from "./board.js";
import { createCluegiverAgent } from "./agents/cluegiver.js";
import { createGuesserAgent } from "./agents/guesser.js";
import { evaluateGuess, countRemaining } from "./game.js";
import { createModelFromEnv } from "./model.js";
import type { BoardState, Clue, GuessResult } from "./types.js";

const N = parseInt(process.argv[2] ?? "10", 10);
const TTFT_TIMEOUT_MS = 15_000;   // max wait for first token
const TOTAL_TIMEOUT_MS = 120_000; // max total time per phase once streaming starts
const MAX_RETRIES = 3;

const model = createModelFromEnv();
const apiKey = process.env.LLM_API_KEY;

interface TurnResult {
  index: number;
  redFound: number;
  redAvailable: number;
  clueWord: string;
  clueNumber: number;
  totalGuesses: number;
  hitAssassin: boolean;
  hitBlue: boolean;
  hitCivilian: boolean;
  stoppedEarly: boolean;
  elapsedMs: number;
  retries: number;
  error?: string;
}

let completed = 0;

function updateStatus() {
  process.stderr.write(`\r\x1b[KDone: ${completed}/${N}`);
}

class TimeoutError extends Error {
  constructor(phase: string, kind: "ttft" | "total", ms: number) {
    const label = kind === "ttft" ? "no first token" : "total time exceeded";
    super(`${phase}: ${label} after ${(ms / 1000).toFixed(0)}s`);
    this.name = "TimeoutError";
  }
}

/**
 * Race a promise against two timeouts:
 *  - ttftMs: max wait for the first token (call `signalFirstToken()` to clear)
 *  - totalMs: max total time from start once streaming begins
 */
function withDualTimeout<T>(
  promise: Promise<T>,
  agent: { subscribe: (fn: (e: any) => void) => () => void; abort: () => void },
  phase: string,
  ttftMs: number,
  totalMs: number,
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    let settled = false;
    let ttftTimer: ReturnType<typeof setTimeout> | null = null;
    let totalTimer: ReturnType<typeof setTimeout> | null = null;

    function cleanup() {
      settled = true;
      if (ttftTimer) clearTimeout(ttftTimer);
      if (totalTimer) clearTimeout(totalTimer);
      unsub();
    }

    function fail(kind: "ttft" | "total", ms: number) {
      if (settled) return;
      cleanup();
      agent.abort();
      reject(new TimeoutError(phase, kind, ms));
    }

    // Start TTFT timer immediately
    ttftTimer = setTimeout(() => fail("ttft", ttftMs), ttftMs);

    // On first token, cancel TTFT timer and start total timer
    const unsub = agent.subscribe((e: any) => {
      if (e.type === "message_start" && ttftTimer) {
        clearTimeout(ttftTimer);
        ttftTimer = null;
        totalTimer = setTimeout(() => fail("total", totalMs), totalMs);
      }
    });

    promise.then(
      (val) => { if (!settled) { cleanup(); resolve(val); } },
      (err) => { if (!settled) { cleanup(); reject(err); } },
    );
  });
}

function runOneTurn(i: number): Promise<TurnResult> {
  return (async () => {
    const t0 = Date.now();
    const board = createRandomMidGameBoard();
    const redAvailable = countRemaining(board, "Red");
    // Snapshot revealed state so we can reset on retry
    const revealedSnapshot = [...board.revealed];

    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      // Reset board state on retry
      if (attempt > 0) {
        for (let j = 0; j < 25; j++) board.revealed[j] = revealedSnapshot[j];
      }

      const cluegiver = createCluegiverAgent(model, apiKey);
      const guesser = createGuesserAgent(model, apiKey);

      try {
        const tClue0 = Date.now();
        const clue = await withDualTimeout(
          cluegiver.getClue(board, 1),
          cluegiver.agent,
          "Cluegiver",
          TTFT_TIMEOUT_MS,
          TOTAL_TIMEOUT_MS,
        );
        const clueMs = Date.now() - tClue0;

        let hitAssassin = false;
        let hitBlue = false;
        let hitCivilian = false;
        let totalGuesses = 0;

        const evaluate = (word: string): GuessResult => {
          const result = evaluateGuess(board, word);
          totalGuesses++;
          if (result.type === "assassin") hitAssassin = true;
          if (result.type === "wrong" && result.color === "Blue") hitBlue = true;
          if (result.type === "wrong" && result.color === "Civilian") hitCivilian = true;
          return result;
        };

        const tGuess0 = Date.now();
        const guessingResult = await withDualTimeout(
          guesser.doGuessing(board, clue, 1, evaluate),
          guesser.agent,
          "Guesser",
          TTFT_TIMEOUT_MS,
          TOTAL_TIMEOUT_MS,
        );
        const guessMs = Date.now() - tGuess0;

        const redFound = redAvailable - countRemaining(board, "Red");
        const elapsedMs = Date.now() - t0;

        completed++;
        process.stderr.write(`\r\x1b[K`);
        const tag = hitAssassin ? "ASSASSIN" : hitBlue ? "BLUE" : hitCivilian ? "CIV" : "OK";
        const retryTag = attempt > 0 ? ` retry=${attempt}` : "";
        console.log(
          `Turn ${i + 1}/${N}: "${clue.word}" for ${clue.number} | Red: ${redFound}/${redAvailable} | Guesses: ${totalGuesses} | ${tag} | total=${(elapsedMs / 1000).toFixed(1)}s clue=${(clueMs / 1000).toFixed(1)}s guess=${(guessMs / 1000).toFixed(1)}s${retryTag}`,
        );
        updateStatus();

        return {
          index: i,
          redFound,
          redAvailable,
          clueWord: clue.word,
          clueNumber: clue.number,
          totalGuesses,
          hitAssassin,
          hitBlue,
          hitCivilian,
          stoppedEarly: guessingResult.stoppedEarly,
          elapsedMs,
          retries: attempt,
        };
      } catch (err) {
        const isTimeout = err instanceof TimeoutError;
        if (isTimeout && attempt < MAX_RETRIES) {
          process.stderr.write(`\r\x1b[K`);
          console.log(`Turn ${i + 1}/${N}: ${(err as TimeoutError).message}, retrying (${attempt + 1}/${MAX_RETRIES})...`);
          continue;
        }

        const elapsedMs = Date.now() - t0;
        completed++;
        process.stderr.write(`\r\x1b[K`);
        const msg = err instanceof Error ? err.message : String(err);
        console.log(`Turn ${i + 1}/${N}: ERROR | ${(elapsedMs / 1000).toFixed(1)}s | ${msg}`);
        updateStatus();

        return {
          index: i,
          redFound: 0,
          redAvailable,
          clueWord: "",
          clueNumber: 0,
          totalGuesses: 0,
          hitAssassin: false,
          hitBlue: false,
          hitCivilian: false,
          stoppedEarly: false,
          elapsedMs,
          retries: attempt,
          error: msg,
        };
      }
    }

    // unreachable, but satisfies TS
    throw new Error("unreachable");
  })();
}

function pct(n: number, total: number): string {
  return `${((n / total) * 100).toFixed(1)}%`;
}

function mean(arr: number[]): number {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function median(arr: number[]): number {
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function printReport(results: TurnResult[]) {
  const valid = results.filter((r) => !r.error);
  const total = results.length;
  const errors = results.filter((r) => r.error);

  console.log("\n" + "=".repeat(60));
  console.log(`RESULTS: ${total} turns (${errors.length} errors)`);
  console.log("=".repeat(60));

  if (valid.length === 0) {
    console.log("\nNo valid turns to report.");
    return;
  }

  const redFoundArr = valid.map((r) => r.redFound);
  const assassinHits = valid.filter((r) => r.hitAssassin);
  const blueHits = valid.filter((r) => r.hitBlue);
  const civHits = valid.filter((r) => r.hitCivilian);
  const stoppedEarly = valid.filter((r) => r.stoppedEarly);
  const guessesArr = valid.map((r) => r.totalGuesses);
  const accuracyArr = valid
    .filter((r) => r.clueNumber > 0)
    .map((r) => r.redFound / r.clueNumber);
  const retriedTurns = valid.filter((r) => r.retries > 0);

  console.log(`\nRed found per turn:`);
  console.log(`  Mean:   ${mean(redFoundArr).toFixed(2)}`);
  console.log(`  Median: ${median(redFoundArr).toFixed(2)}`);

  console.log(`\nSafety:`);
  console.log(`  Assassin hit rate: ${assassinHits.length}/${valid.length} (${pct(assassinHits.length, valid.length)})`);
  console.log(`  Blue hit rate:     ${blueHits.length}/${valid.length} (${pct(blueHits.length, valid.length)})`);
  console.log(`  Civilian hit rate: ${civHits.length}/${valid.length} (${pct(civHits.length, valid.length)})`);
  console.log(`  Stop-early rate:   ${stoppedEarly.length}/${valid.length} (${pct(stoppedEarly.length, valid.length)})`);

  console.log(`\nEfficiency:`);
  console.log(`  Mean accuracy (redFound/clueNumber): ${accuracyArr.length > 0 ? mean(accuracyArr).toFixed(2) : "N/A"}`);
  console.log(`  Mean guesses per turn:               ${mean(guessesArr).toFixed(2)}`);
  console.log(`  Turns retried:                       ${retriedTurns.length}/${valid.length} (${pct(retriedTurns.length, valid.length)})`);

  console.log(`\nPer-turn breakdown:`);
  console.log(
    `  ${"#".padEnd(4)} ${"Clue".padEnd(16)} ${"Num".padEnd(4)} ${"Red".padEnd(8)} ${"Guess".padEnd(6)} ${"Result".padEnd(10)} ${"Time".padEnd(8)} ${"Retry".padEnd(5)}`,
  );
  console.log(
    `  ${"-".repeat(4)} ${"-".repeat(16)} ${"-".repeat(4)} ${"-".repeat(8)} ${"-".repeat(6)} ${"-".repeat(10)} ${"-".repeat(8)} ${"-".repeat(5)}`,
  );
  for (const r of results) {
    if (r.error) {
      console.log(`  ${String(r.index + 1).padEnd(4)} ${"ERROR".padEnd(16)} ${"-".padEnd(4)} ${"-".padEnd(8)} ${"-".padEnd(6)} ${"ERROR".padEnd(10)} ${(r.elapsedMs / 1000).toFixed(1).padEnd(8)} ${String(r.retries).padEnd(5)}`);
      continue;
    }
    const tag = r.hitAssassin ? "ASSASSIN" : r.hitBlue ? "BLUE" : r.hitCivilian ? "CIVILIAN" : r.stoppedEarly ? "STOPPED" : "CLEAN";
    console.log(
      `  ${String(r.index + 1).padEnd(4)} ${r.clueWord.padEnd(16)} ${String(r.clueNumber).padEnd(4)} ${`${r.redFound}/${r.redAvailable}`.padEnd(8)} ${String(r.totalGuesses).padEnd(6)} ${tag.padEnd(10)} ${(r.elapsedMs / 1000).toFixed(1).padEnd(8)} ${String(r.retries).padEnd(5)}`,
    );
  }
}

async function main() {
  console.log(`Running ${N} turns in parallel with model: ${model.id}`);
  console.log(`Timeout: ${TTFT_TIMEOUT_MS / 1000}s TTFT / ${TOTAL_TIMEOUT_MS / 1000}s total, max ${MAX_RETRIES} retries`);
  console.log(`Base URL: ${model.baseUrl}\n`);

  const promises = Array.from({ length: N }, (_, i) => runOneTurn(i));
  const results = await Promise.all(promises);

  results.sort((a, b) => a.index - b.index);
  process.stderr.write(`\r\x1b[K`);
  printReport(results);
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
