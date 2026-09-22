/** UI-only state for the text/voice -> gloss translation flow (Phase 14) --
 * not part of the wire protocol, see types/api.ts TextToGlossResponse for that. */
export interface GlossLabel {
  text: string;
  isFingerspell: boolean;
}

export type TranslationState =
  | { status: "idle" }
  | { status: "loading" }
  | {
      status: "success";
      glossSequence: string[];
      /** Ukrainian word per gloss token -- see types/api.ts's
       * TextToGlossResponse.gloss_labels; never show glossSequence's own
       * tokens directly, they're internal identifiers, not Ukrainian. */
      glossLabels: GlossLabel[];
      /** Full composed Ukrainian sentence, or null if this sequence
       * doesn't match a supported grammatical pattern. */
      composedText: string | null;
    }
  | { status: "error"; message: string };
