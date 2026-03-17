import { Agent } from "@mariozechner/pi-agent-core";
import type { AgentTool } from "@mariozechner/pi-agent-core";
import { Type, streamSimple } from "@mariozechner/pi-ai";
import type { Model } from "@mariozechner/pi-ai";
import type { BoardState, Clue } from "../types.js";
import { formatBoardForCluegiver } from "./shared.js";

const CLUEGIVER_SYSTEM_PROMPT = `You are a Codenames cluegiver (codemaster). You are playing cooperatively with a guesser to find all RED words on the board.

RULES:
- You can see the board AND the secret key grid showing which words are Red, Blue, Civilian, or Assassin.
- Your goal: give a one-word clue and a number indicating how many board words relate to that clue.
- The guesser will try to guess the Red words based on your clue.
- Your clue must be a SINGLE word (no spaces, no hyphens, no parts of board words).
- The number should indicate how many Red words your clue relates to.
- AVOID clues that might lead the guesser to Blue, Civilian, or especially the Assassin word.
- Think carefully about which words could be confused before giving your clue.

STRATEGY:
- Start by trying to connect 2-3 Red words with a clue to make progress efficiently.
- Avoid clues that could also match Blue or Assassin words.
- If only 1 Red word remains, give a very specific clue with number 1.
- Consider what the guesser might think - avoid ambiguous clues.

You MUST use the give_clue tool to submit your clue. Do not just write the clue in text.`;

const clueParams = Type.Object({
  clue: Type.String({ description: "A single-word clue (no spaces, no hyphens, no board words)" }),
  number: Type.Integer({ description: "How many Red words this clue relates to (1-8)", minimum: 1, maximum: 8 }),
});

export function createCluegiverAgent(model: Model<any>, apiKey?: string) {
  let capturedClue: Clue | null = null;

  const giveClue: AgentTool<typeof clueParams> = {
    name: "give_clue",
    label: "Give Clue",
    description: "Submit your clue to the guesser. The clue must be a single word (no spaces, no hyphens). The number indicates how many words on the board relate to your clue.",
    parameters: clueParams,
    execute: async (_toolCallId, params) => {
      const word = params.clue.toUpperCase().trim();
      if (word.includes(" ") || word.includes("-")) {
        throw new Error("Clue must be a single word with no spaces or hyphens.");
      }
      capturedClue = { word, number: params.number };
      return {
        content: [{ type: "text" as const, text: `Clue submitted: "${capturedClue.word}" for ${capturedClue.number}` }],
        details: { clue: capturedClue },
      };
    },
  };

  const agent = new Agent({
    initialState: {
      systemPrompt: CLUEGIVER_SYSTEM_PROMPT,
      model,
      tools: [giveClue],
    },
    streamFn: streamSimple,
    getApiKey: apiKey ? () => apiKey : undefined,
  });

  async function getClue(board: BoardState, turnNumber: number): Promise<Clue> {
    capturedClue = null;
    agent.clearMessages();

    const boardView = formatBoardForCluegiver(board);
    const prompt = `Turn ${turnNumber}. Here is the current board state:\n\n${boardView}\n\nGive your clue using the give_clue tool.`;

    await agent.prompt(prompt);

    if (!capturedClue) {
      throw new Error("Cluegiver agent did not call the give_clue tool.");
    }
    return capturedClue;
  }

  return { agent, getClue };
}
