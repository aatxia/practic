/**
 * Ukrainian display label per gloss token, for surfaces that only have a
 * raw gloss sequence to work with (the 3D avatar's live status caption and
 * "no animation for" list -- components/Avatar/Avatar.tsx), not the
 * richer gloss_labels the backend now returns for a text-to-gloss
 * request (components/Transcript/Transcript.tsx uses that instead).
 *
 * Mirrors ml/nlp/lexicon.py's PRONOUNS/VERBS/NOUNS/ADVERBS/STANDALONE
 * surface forms (pronoun lemma, verb infinitive, noun nominative) -- kept
 * here as static data, not fetched, because the avatar's per-frame render
 * loop needs a synchronous lookup. gloss_sequence's own tokens (e.g.
 * "WANT", "CAR") are internal English identifiers, never meant to be read
 * by a user -- this table is what stands in for them.
 */

/** Grammar marker gloss with no Ukrainian surface form of its own (see
 * lexicon.py's module docstring) -- the 3D avatar (components/Avatar/)
 * has no hand-authored pose for it, so it's filtered out of that display
 * specifically. It DOES have a real reference video (components/
 * GestureVideo/gestureVideos.ts), so that component shows it like any
 * other gloss rather than reusing this filtering. */
export const TENSE_PAST_GLOSS = "PAST";

const UKRAINIAN_LABELS: Record<string, string> = {
  // Pronouns (lemma)
  I: "я",
  YOU: "ти",
  HE: "він",
  SHE: "вона",
  WE: "ми",
  YOU_PL: "ви",
  THEY: "вони",
  // Verbs (infinitive)
  WANT: "хотіти",
  LIKE: "любити",
  HAVE: "мати",
  KNOW: "знати",
  UNDERSTAND: "розуміти",
  SEE: "бачити",
  READ: "читати",
  WRITE: "писати",
  EAT: "їсти",
  DRINK: "пити",
  GO: "іти",
  WORK: "працювати",
  LIVE: "жити",
  SLEEP: "спати",
  SPEAK: "говорити",
  RIDE: "їхати",
  NEED: "потребувати",
  HELP: "допомагати",
  // Nouns (nominative)
  WATER: "вода",
  FOOD: "їжа",
  BREAD: "хліб",
  TEA: "чай",
  COFFEE: "кава",
  MILK: "молоко",
  JUICE: "сік",
  SOUP: "суп",
  MEAT: "м'ясо",
  SUGAR: "цукор",
  BOOK: "книга",
  FLOWER: "квітка",
  LETTER: "лист",
  PHONE: "телефон",
  HOUSE: "будинок",
  CAR: "машина",
  WORK_N: "робота",
  SCHOOL: "школа",
  CITY: "місто",
  FRIEND: "друг",
  NAME: "ім'я",
  TIME: "час",
  DAY: "день",
  COUNTRY: "країна",
  FAMILY: "сім'я",
  MONEY: "гроші",
  // Adverbs
  YESTERDAY: "вчора",
  TODAY: "сьогодні",
  TOMORROW: "завтра",
  MORNING: "вранці",
  EVENING: "ввечері",
  NIGHT: "вночі",
  NOW: "зараз",
  HOME: "додому",
  // Standalone phrases
  PRIVIT: "привіт",
  DYAKUYU: "дякую",
  TAK: "так",
  NI: "ні",
  BUD_LASKA: "будь ласка",
  DOBRANICH: "добраніч",
  VYBACHTE: "вибачте",
  DO_POBACHENNYA: "до побачення",
  // Negation particle
  NOT: "не",
  // Grammar marker (see TENSE_PAST_GLOSS below) -- filtered out of the 3D
  // avatar's display (it has no pose for it), but GestureVideo.tsx does
  // have a real reference video for it, so it needs a real label here too.
  PAST: "минулий час",
};

/** Ukrainian label for a single gloss token, or the token itself (with
 * underscores turned into spaces, purely cosmetic) if it isn't in the
 * table -- an honest fallback for an unmapped gloss, never a guess at a
 * translation. */
export function glossLabel(gloss: string): string {
  return UKRAINIAN_LABELS[gloss] ?? gloss.replace(/_/g, " ");
}
