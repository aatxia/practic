import { describe, expect, it } from "vitest";
import { GLOSS_POSES, NEUTRAL_POSE, poseForGloss } from "./poses";

describe("poseForGloss", () => {
  it("returns a pose for each of the 5 demo-dataset glosses", () => {
    for (const gloss of ["PRIVIT", "DYAKUYU", "TAK", "NI", "BUD_LASKA"]) {
      expect(poseForGloss(gloss)).not.toBeNull();
    }
  });

  it("returns null for a gloss with no defined animation, rather than guessing", () => {
    expect(poseForGloss("UNKNOWN_GLOSS")).toBeNull();
    expect(poseForGloss("WATER")).toBeNull();
  });

  it("every defined pose differs from the neutral pose", () => {
    for (const pose of Object.values(GLOSS_POSES)) {
      expect(pose.target).not.toEqual(NEUTRAL_POSE);
    }
  });
});
