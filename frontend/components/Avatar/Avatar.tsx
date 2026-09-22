"use client";

import { Info, Play, RotateCcw } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { formatGlossLabel, groupGlossesForDisplay } from "@/lib/glossDisplay";
import { TENSE_PAST_GLOSS } from "@/lib/glossLabels";
import { GlossPlayer } from "./player";
import { poseForGloss } from "./poses";
import { applyRotations, buildPuppet } from "./puppet";
import { applyRiggedRotations, loadRiggedPuppet, type RiggedPuppet } from "./riggedPuppet";
import { riggedPoseForGloss, RIG_NEUTRAL_POSE } from "./riggedPoses";

interface AvatarProps {
  /** Confirmed gloss sequence to animate, in order. Empty = idle/neutral. */
  glossSequence: string[];
}

interface PlaybackStatus {
  currentGloss: string | null;
  unanimatedGlosses: string[];
}

const IDLE_STATUS: PlaybackStatus = { currentGloss: null, unanimatedGlosses: [] };

const SPEED_OPTIONS = [0.5, 1, 1.5, 2] as const;

/** Detected via a throwaway canvas, not the real one -- cheap and synchronous,
 * jsdom and a real browser without WebGL both fail this the same way. Must
 * still run from an effect, not a useState lazy initializer: that runs
 * during SSR too, where `document`/`window` are undefined and this always
 * returns false -- if the client's real (and most likely true) answer were
 * read synchronously on the very first client render, that render would
 * already show different markup (the real canvas vs. the SSR fallback
 * message below) than what the server sent, a real hydration mismatch
 * reported live. */
function detectWebglSupport(): boolean {
  if (typeof document === "undefined" || typeof window === "undefined") {
    return false;
  }
  try {
    const canvas = document.createElement("canvas");
    return !!(
      window.WebGLRenderingContext &&
      (canvas.getContext("webgl2") ?? canvas.getContext("webgl"))
    );
  } catch {
    return false;
  }
}

export function Avatar({ glossSequence }: AvatarProps): React.ReactElement {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const playerRef = useRef<GlossPlayer>(new GlossPlayer());
  const glossSequenceRef = useRef<string[]>(glossSequence);
  const speedRef = useRef(1);
  // Starts true (matching what the server always renders, since it never
  // has a real `window`/`document` to detect WebGL with) and is corrected
  // in an effect below -- see detectWebglSupport's docstring for why a
  // useState lazy initializer isn't safe here.
  const [webglUnsupported, setWebglUnsupported] = useState(true);
  const [status, setStatus] = useState<PlaybackStatus>(IDLE_STATUS);
  const [speed, setSpeed] = useState(1);
  const [usingRiggedModel, setUsingRiggedModel] = useState(false);

  useEffect(() => {
    setWebglUnsupported(!detectWebglSupport());
  }, []);

  useEffect(() => {
    speedRef.current = speed;
  }, [speed]);

  // TENSE_PAST_GLOSS ("PAST") is a pure grammar marker (ml/nlp/lexicon.py)
  // with no gesture of its own -- feeding it to GlossPlayer would hold a
  // pointless neutral pose for a full HOLD_SECONDS and misreport it as a
  // "missing animation", so it's dropped before the player ever sees it.
  const animatableGlossSequence = useMemo(
    () => glossSequence.filter((gloss) => gloss !== TENSE_PAST_GLOSS),
    [glossSequence],
  );

  useEffect(() => {
    glossSequenceRef.current = animatableGlossSequence;
    playerRef.current.play(animatableGlossSequence);
  }, [animatableGlossSequence]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || webglUnsupported) {
      return;
    }

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    } catch (err) {
      // Already gated by detectWebglSupport() above; a failure here despite
      // that positive detection is a rare edge case, not worth a second
      // render pass to surface in the UI -- fail silently, no fake canvas.
      console.error("THREE.WebGLRenderer failed to initialize:", err);
      return;
    }

    const width = canvas.clientWidth || 320;
    const height = canvas.clientHeight || 224;
    renderer.setSize(width, height, false);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(35, width / height, 0.1, 10);

    // Three-point lighting for contrast: a bright key light casting real
    // shadow-side definition, a dim cool fill so the shadow side isn't pure
    // black, and a rim light behind the puppet to separate it from the dark
    // viewport background.
    scene.add(new THREE.AmbientLight(0xffffff, 0.25));
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.4);
    keyLight.position.set(2, 3, 2);
    scene.add(keyLight);
    const fillLight = new THREE.DirectionalLight(0x8ab4ff, 0.35);
    fillLight.position.set(-2.5, 1, 1.5);
    scene.add(fillLight);
    const rimLight = new THREE.DirectionalLight(0xffffff, 0.6);
    rimLight.position.set(0, 2, -3);
    scene.add(rimLight);

    const proceduralPuppet = buildPuppet();
    scene.add(proceduralPuppet.root);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.enablePan = false;

    // Puppet now has legs and hair reaching from ~y=0.03 (feet) to ~y=2.43
    // (hair top) -- the camera targets its vertical center and sits far
    // enough back for that full ~2.4-unit height to stay in frame.
    function frameForProceduralPuppet(): void {
      camera.position.set(0, 1.25, 4.6);
      controls.target.set(0, 1.2, 0);
      controls.minDistance = 2.4;
      controls.maxDistance = 7;
      camera.lookAt(controls.target);
      controls.update();
    }

    // The rigged model (see riggedPuppet.ts) is a smaller, ~1.5-unit-tall
    // figure with feet at y=0 -- measured from its own bounding box, not
    // guessed, since it's a completely different asset with its own scale.
    function frameForRiggedPuppet(): void {
      camera.position.set(0, 0.9, 3.3);
      controls.target.set(0, 0.8, 0);
      controls.minDistance = 1.6;
      controls.maxDistance = 5;
      camera.lookAt(controls.target);
      controls.update();
    }

    frameForProceduralPuppet();

    let riggedPuppet: RiggedPuppet | null = null;
    let cancelled = false;
    loadRiggedPuppet()
      .then((loaded) => {
        if (cancelled) {
          return;
        }
        scene.remove(proceduralPuppet.root);
        scene.add(loaded.root);
        riggedPuppet = loaded;
        frameForRiggedPuppet();
        // A different pose table (riggedPoses.ts) and neutral pose than the
        // procedural puppet's, but the exact same GlossPlayer sequencing/
        // transition/wobble logic -- see player.ts's constructor docstring.
        playerRef.current = new GlossPlayer(RIG_NEUTRAL_POSE, riggedPoseForGloss);
        playerRef.current.play(glossSequenceRef.current);
        setUsingRiggedModel(true);
      })
      .catch((err: unknown) => {
        // Honest fallback, same principle as puppet.ts's "no defined pose
        // -> hold neutral, don't guess": a slow network, a blocked asset,
        // or an unexpected glTF shape never leaves the avatar blank or
        // broken, it just keeps the procedural puppet already on screen.
        console.warn("Rigged avatar model unavailable, staying on the procedural puppet:", err);
      });

    let frameId: number;
    let lastTime = performance.now();

    function tick(now: number): void {
      const delta = Math.min((now - lastTime) / 1000, 0.1);
      lastTime = now;

      const frame = playerRef.current.update(delta * speedRef.current);
      if (riggedPuppet) {
        applyRiggedRotations(riggedPuppet, frame.rotations);
      } else {
        applyRotations(proceduralPuppet, frame.rotations);
      }
      setStatus({ currentGloss: frame.currentGloss, unanimatedGlosses: frame.unanimatedGlosses });

      controls.update();
      renderer.render(scene, camera);
      frameId = requestAnimationFrame(tick);
    }
    frameId = requestAnimationFrame(tick);

    return () => {
      cancelled = true;
      cancelAnimationFrame(frameId);
      controls.dispose();
      renderer.dispose();
    };
  }, [webglUnsupported]);

  if (webglUnsupported) {
    return (
      <div className="flex h-40 items-center justify-center rounded-xl bg-slate-100 px-4 text-center text-sm text-slate-400">
        WebGL не підтримується цим браузером — 3D avatar недоступний.
      </div>
    );
  }

  const displayItems = groupGlossesForDisplay(glossSequence);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between text-xs">
        <span className="inline-flex items-center gap-1 rounded-full border border-slate-200 px-2 py-0.5 font-medium text-slate-500">
          <Info className="h-3 w-3" aria-hidden />
          Демо-жести, не справжня УЖМ
        </span>
        {status.currentGloss && (
          <span className="text-slate-500">{formatGlossLabel(status.currentGloss)}</span>
        )}
      </div>

      <canvas ref={canvasRef} className="h-72 w-full rounded-xl bg-slate-800" />
      <p className="text-center text-[11px] text-slate-400">
        Перетягніть, щоб обертати модель — прокрутіть, щоб наблизити
        {usingRiggedModel && ' · Модель: "Cesium Man" (CC BY 4.0, cesium.com)'}
      </p>

      <div className="flex items-center gap-2 text-xs">
        <span className="font-medium text-slate-500">Швидкість:</span>
        <div className="flex gap-1">
          {SPEED_OPTIONS.map((option) => (
            <button
              key={option}
              type="button"
              onClick={() => setSpeed(option)}
              aria-pressed={speed === option}
              className={`rounded-full px-2 py-0.5 font-medium transition-colors ${
                speed === option
                  ? "bg-brand-600 text-white"
                  : "bg-slate-100 text-slate-500 hover:bg-slate-200"
              }`}
            >
              {option}×
            </button>
          ))}
        </div>
      </div>

      {displayItems.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5">
          <button
            type="button"
            onClick={() => playerRef.current.play(animatableGlossSequence)}
            title="Переглянути всі жести один за одним"
            className="inline-flex items-center gap-1 rounded-full bg-brand-600 px-2.5 py-0.5 text-xs font-medium text-white hover:bg-brand-700"
          >
            <Play className="h-3 w-3" aria-hidden />
            Переглянути все
          </button>
          {displayItems.map((item) => {
            // riggedPoses.ts's RIGGED_GLOSS_POSES covers exactly the same
            // gloss keys as poses.ts's GLOSS_POSES (see its module
            // docstring), so this check is valid regardless of which
            // backend is currently active.
            const canReplay = item.tokens.some((token) => poseForGloss(token) !== null);
            return (
              <button
                key={item.key}
                type="button"
                disabled={!canReplay}
                onClick={() => playerRef.current.play(item.tokens)}
                title={canReplay ? `Повторити жест «${item.label}»` : "Немає анімації для цього жесту"}
                className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium ${
                  canReplay
                    ? "border-slate-200 text-slate-600 hover:border-brand-300 hover:bg-brand-50 hover:text-brand-700"
                    : "cursor-not-allowed border-slate-100 text-slate-300"
                }`}
              >
                {canReplay && <RotateCcw className="h-3 w-3" aria-hidden />}
                {item.label}
              </button>
            );
          })}
        </div>
      )}

      {status.unanimatedGlosses.length > 0 && (
        <p className="text-xs text-slate-400">
          Немає анімації для:{" "}
          {groupGlossesForDisplay(status.unanimatedGlosses)
            .map((item) => item.label)
            .join(", ")}
        </p>
      )}
    </div>
  );
}
