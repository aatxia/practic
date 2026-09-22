export type SpeechStatus = "idle" | "listening" | "stopped" | "error";

export type SpeechErrorReason =
  | "unsupported"
  | "permission_denied"
  | "no_speech"
  | "network"
  | "unknown";

export interface SpeechError {
  reason: SpeechErrorReason;
  message: string;
}
