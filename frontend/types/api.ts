export interface HealthResponse {
  status: string;
  app_env: string;
  model_type: string;
  features: {
    hands: boolean;
    pose: boolean;
    face: boolean;
  };
  ml_pipeline_status: "not_implemented" | "demo_mode" | "ready";
}

// --- WebSocket protocol types (implemented in Phase 5) ---
// Declared now so components built in later phases share one source of truth.

export interface FrameMessage {
  type: "frame";
  timestamp: number;
  data: string; // base64-encoded JPEG
}

/** Non-manual grammar marker (Phase 17): eyebrow position relative to the
 * signer's own calibrated neutral face. "NONE" also covers "no face
 * detected" and "still calibrating" -- see ml/features/facial_grammar.py. */
export type FacialGrammarMarker = "NONE" | "EYEBROWS_RAISED" | "EYEBROWS_FURROWED";

/** One ranked alternative from the word classifier's real top-k output --
 * never an invented or "similar-sounding" guess (backend/websocket/
 * protocol.py's WordCandidate). */
export interface WordCandidate {
  gloss: string;
  confidence: number;
}

export interface PredictionMessage {
  type: "prediction" | "final_prediction";
  text: string;
  /** Raw predicted sign label for this frame (Phase 15: drives the avatar --
   * only consumed once is_final confirms it, never an interim guess). */
  gloss: string;
  confidence: number;
  is_final: boolean;
  facial_grammar: FacialGrammarMarker;
  /** The model's real ranked alternatives -- present ONLY when is_final is
   * true (a manual-correction control needs this for a word that was just
   * CONFIRMED and saved, not for a still-changing interim guess). null
   * otherwise, never a fabricated or empty-but-present list. */
  top_k: WordCandidate[] | null;
}

/** A single (x, y) point, image-normalized to [0, 1] in MediaPipe's own
 * frame space (not the centered/scaled coordinates used for ML features) --
 * suitable for drawing directly on the video frame. */
export type LandmarkPoint = [number, number];

/** Real per-frame detection (ml/preprocessing/normalization.py's `present`
 * dict), sent for every processed frame independent of ML readiness --
 * drives the opt-in "is my hand visible" indicator from frame 1. */
export interface LandmarksStatusMessage {
  type: "landmarks_status";
  left_hand: boolean;
  right_hand: boolean;
  pose: boolean;
  face: boolean;
  /** Raw (x, y) points for the overlay (frontend/components/Camera) --
   * null when that modality wasn't detected this frame. Face is
   * intentionally omitted (see backend/websocket/protocol.py). */
  left_hand_points: LandmarkPoint[] | null;
  right_hand_points: LandmarkPoint[] | null;
  pose_points: LandmarkPoint[] | null;
}

/** A single dactyl letter from the SEPARATE real fingerspelling classifier
 * (ml/fingerspelling/, trained on real photos -- not the synthetic-only
 * word-level checkpoint PredictionMessage comes from). Sent only when
 * exactly one hand is detected and that checkpoint is loaded -- silently
 * absent otherwise, same honesty convention as landmarks_status. */
/** Mirrors WordCandidate for the fingerspelling classifier's real top-k
 * output (backend/websocket/protocol.py's LetterCandidate). */
export interface LetterCandidate {
  letter: string;
  confidence: number;
}

export interface LetterPredictionMessage {
  type: "letter_prediction" | "letter_confirmed";
  letter: string;
  confidence: number;
  is_final: boolean;
  /** Present only when is_final is true -- see WordCandidate's docstring,
   * same reasoning for the letter classifier. */
  top_k: LetterCandidate[] | null;
}

/** Continuous-dictation boundary signal (backend/websocket/protocol.py):
 * fires from a sustained no-hand streak, exactly once per threshold
 * crossing. "token" means whatever was mid-flight (mainly an in-progress
 * fingerspelled word) is done and should be finalized into one token;
 * "utterance" means the whole utterance is done and language
 * post-processing should run now. "token" always fires before (or
 * alongside) "utterance" for the same pause, never after. */
export interface BoundaryMessage {
  type: "boundary";
  kind: "token" | "utterance";
}

export interface ErrorMessage {
  type: "error";
  message: string;
}

export interface ConnectionMessage {
  type: "connection";
  status: "ok" | "closed";
}

export type ServerMessage =
  | PredictionMessage
  | LandmarksStatusMessage
  | LetterPredictionMessage
  | BoundaryMessage
  | ErrorMessage
  | ConnectionMessage;

// --- REST API types (Phase 14) ---

export interface TextToGlossRequest {
  text: string;
}

export interface GlossLabel {
  text: string;
  is_fingerspell: boolean;
}

export interface TextToGlossResponse {
  gloss_sequence: string[];
  /** Ukrainian word/phrase per gloss token -- gloss_sequence's own tokens
   * (e.g. "WANT", "CAR") are internal English identifiers, never meant to
   * be shown to a user. */
  gloss_labels: GlossLabel[];
  /** Full composed Ukrainian sentence, when the sequence matches a
   * supported grammatical pattern; null otherwise (gloss_labels still
   * shows what was understood, word by word). */
  composed_text: string | null;
}

/** Reverse direction of text-to-gloss (Phase 20): composes a full sentence
 * from glosses accumulated across several live-camera confirmations. */
export interface GlossToTextRequest {
  gloss_sequence: string[];
}

export interface GlossToTextResponse {
  gloss_labels: GlossLabel[];
  /** Full composed Ukrainian sentence, from whichever engine
   * composed_text_source names -- null when neither could. */
  composed_text: string | null;
  /** "rule_based" is grammar-guaranteed within ml/nlp/gloss_to_text.py's
   * coverage; "ai_fallback" is a real LLM's best attempt (backend/app/
   * services/ai_sentence_composer.py), only ever consulted when
   * rule_based couldn't compose a pattern -- must be shown to the user as
   * AI-generated, not verified, never with the same confidence as
   * rule_based. null when composed_text itself is null. */
  composed_text_source: "rule_based" | "ai_fallback" | null;
}
