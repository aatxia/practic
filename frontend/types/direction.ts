/** Which side is input and which is output in the translator's single
 * combined interface (see components/Translator/DirectionSwitch.tsx) --
 * switching this swaps the active panel in place, no page reload/route
 * change. */
export type TranslationDirection = "gestures-to-text" | "text-to-gestures";
