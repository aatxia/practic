import type { GlossToTextResponse, HealthResponse, TextToGlossResponse } from "@/types/api";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {}

export async function fetchHealth(): Promise<HealthResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}/health`, { cache: "no-store" });
  } catch (cause) {
    throw new ApiError(
      `Не вдалося з'єднатися з backend на ${API_URL}. Переконайся, що ` +
        `\`uvicorn app.main:app --reload\` запущено.`,
      { cause },
    );
  }

  if (!response.ok) {
    throw new ApiError(`Backend health check failed: HTTP ${response.status}`);
  }

  return (await response.json()) as HealthResponse;
}

/** Phase 14: Ukrainian text -> gloss sequence, via the rule-based backend
 * engine (ml/nlp/text_to_gloss.py). Throws ApiError with the backend's own
 * message (e.g. naming an unrecognized word) on a 4xx/5xx response. */
export async function textToGloss(text: string): Promise<TextToGlossResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}/translate/text-to-gloss`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
  } catch (cause) {
    throw new ApiError(`Не вдалося з'єднатися з backend на ${API_URL}.`, { cause });
  }

  const body: unknown = await response.json().catch(() => null);

  if (!response.ok) {
    const detail =
      body && typeof body === "object" && "detail" in body && typeof body.detail === "string"
        ? body.detail
        : undefined;
    throw new ApiError(detail ?? `Переклад не вдався: HTTP ${response.status}`);
  }

  return body as TextToGlossResponse;
}

/** Composes a full Ukrainian sentence from an accumulated sequence of
 * live-camera-confirmed glosses (gathered from final_prediction messages
 * while the signer keeps going, posted once they press "Зупинити"). Same
 * rule-based engine as textToGloss, just the reverse direction. */
export async function glossToText(glossSequence: string[]): Promise<GlossToTextResponse> {
  let response: Response;
  try {
    response = await fetch(`${API_URL}/translate/gloss-to-text`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ gloss_sequence: glossSequence }),
    });
  } catch (cause) {
    throw new ApiError(`Не вдалося з'єднатися з backend на ${API_URL}.`, { cause });
  }

  const body: unknown = await response.json().catch(() => null);

  if (!response.ok) {
    const detail =
      body && typeof body === "object" && "detail" in body && typeof body.detail === "string"
        ? body.detail
        : undefined;
    throw new ApiError(detail ?? `Переклад не вдався: HTTP ${response.status}`);
  }

  return body as GlossToTextResponse;
}
