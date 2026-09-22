/**
 * Hero photo (public/hero.png, uploaded by the signer herself) -- replaces
 * the earlier hand-drawn line-art placeholder that existed only because no
 * real source image was available to embed at the time.
 */
export function SignToSpeech(): React.ReactElement {
  return (
    // A static local asset with no responsive/format needs next/image's
    // build-time optimization would add; plain <img> avoids that machinery.
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src="/hero.png"
      alt="Жестова мова стає текстом і голосом"
      className="h-full w-full rounded-2xl object-cover"
    />
  );
}
