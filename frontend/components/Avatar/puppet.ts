/**
 * Procedural primitive-based humanoid puppet for the placeholder 3D avatar
 * (Phase 15, reworked). Still no rigged/skinned GLTF asset (waiting on an
 * external model file, see PROJECT_STATUS.md) -- but capsule-based limbs,
 * actual fingered hands, legs, and simple hair/eyes read as a recognizable
 * figure instead of floating cylinders, parented into the same joint-pivot
 * `Object3D` hierarchy so poses.ts's rotations still drive it directly.
 *
 * Pure scene-graph construction: builds `THREE.Object3D`s but never touches
 * a canvas/WebGL context, so it's unit-testable without a real GPU (unlike
 * `THREE.WebGLRenderer`, which Avatar.tsx wraps separately with a fallback).
 */
import * as THREE from "three";
import type { JointRotations } from "./poses";

export interface Puppet {
  root: THREE.Group;
  head: THREE.Object3D;
  leftShoulder: THREE.Object3D;
  leftElbow: THREE.Object3D;
  rightShoulder: THREE.Object3D;
  rightElbow: THREE.Object3D;
}

interface ArmMaterials {
  sleeve: THREE.Material;
  skin: THREE.Material;
}

/** A relaxed, slightly curled hand -- four fingers plus a thumb -- rigidly
 * attached at the wrist (end of the forearm). No separate wrist/finger
 * joints exist in poses.ts yet, so the hand doesn't animate on its own; it
 * moves as one piece with the forearm, which is enough to read as a hand
 * during a shoulder/elbow gesture rather than a bare rod. */
function buildHand(sign: 1 | -1, skin: THREE.Material): THREE.Group {
  const hand = new THREE.Group();

  const palm = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.11, 0.04), skin);
  palm.position.y = -0.07;
  hand.add(palm);

  const fingerGeometry = new THREE.CapsuleGeometry(0.013, 0.07, 4, 6);
  for (let i = 0; i < 4; i += 1) {
    const finger = new THREE.Mesh(fingerGeometry, skin);
    finger.position.set(-0.033 + i * 0.022, -0.16, 0);
    finger.rotation.z = (i - 1.5) * 0.05;
    hand.add(finger);
  }

  const thumb = new THREE.Mesh(new THREE.CapsuleGeometry(0.014, 0.05, 4, 6), skin);
  thumb.position.set(sign * 0.065, -0.09, 0.01);
  thumb.rotation.z = sign * 0.9;
  hand.add(thumb);

  return hand;
}

function buildArm(sign: 1 | -1, materials: ArmMaterials): { shoulder: THREE.Group; elbow: THREE.Group } {
  const shoulder = new THREE.Group();
  shoulder.position.set(sign * 0.42, 1.02, 0);

  const upperArm = new THREE.Mesh(new THREE.CapsuleGeometry(0.075, 0.34, 4, 10), materials.sleeve);
  upperArm.position.y = -0.24;
  shoulder.add(upperArm);

  const elbow = new THREE.Group();
  elbow.position.y = -0.48;
  shoulder.add(elbow);

  const forearm = new THREE.Mesh(new THREE.CapsuleGeometry(0.06, 0.3, 4, 10), materials.skin);
  forearm.position.y = -0.21;
  elbow.add(forearm);

  const hand = buildHand(sign, materials.skin);
  hand.position.y = -0.42;
  elbow.add(hand);

  return { shoulder, elbow };
}

function buildLeg(sign: 1 | -1, pantsMaterial: THREE.Material, shoeMaterial: THREE.Material): THREE.Group {
  const leg = new THREE.Group();
  leg.position.set(sign * 0.16, 0.02, 0);

  const upperLeg = new THREE.Mesh(new THREE.CapsuleGeometry(0.1, 0.32, 4, 10), pantsMaterial);
  upperLeg.position.y = -0.21;
  leg.add(upperLeg);

  const lowerLeg = new THREE.Mesh(new THREE.CapsuleGeometry(0.08, 0.3, 4, 10), pantsMaterial);
  lowerLeg.position.y = -0.52;
  leg.add(lowerLeg);

  const shoe = new THREE.Mesh(new THREE.BoxGeometry(0.13, 0.07, 0.2), shoeMaterial);
  shoe.position.set(0, -0.71, 0.05);
  leg.add(shoe);

  return leg;
}

export function buildPuppet(): Puppet {
  const root = new THREE.Group();

  const jacketMaterial = new THREE.MeshStandardMaterial({ color: 0xdc2626, roughness: 0.65 });
  const skinMaterial = new THREE.MeshStandardMaterial({ color: 0xffd9b8, roughness: 0.6 });
  const hairMaterial = new THREE.MeshStandardMaterial({ color: 0x27272a, roughness: 0.5 });
  const pantsMaterial = new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.7 });
  const shoeMaterial = new THREE.MeshStandardMaterial({ color: 0x18181b, roughness: 0.5 });
  const eyeMaterial = new THREE.MeshStandardMaterial({ color: 0x18181b, roughness: 0.3 });

  const legs = new THREE.Group();
  legs.add(buildLeg(-1, pantsMaterial, shoeMaterial), buildLeg(1, pantsMaterial, shoeMaterial));
  root.add(legs);

  const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.32, 0.46, 4, 12), jacketMaterial);
  torso.position.y = 0.68;
  root.add(torso);

  const neck = new THREE.Mesh(new THREE.CapsuleGeometry(0.09, 0.04, 4, 8), skinMaterial);
  neck.position.y = 1.0;
  root.add(neck);

  const headPivot = new THREE.Group();
  headPivot.position.set(0, 1.1, 0);
  const headMesh = new THREE.Mesh(new THREE.SphereGeometry(0.26, 20, 20), skinMaterial);
  headMesh.position.y = 0.26;
  headPivot.add(headMesh);

  const hair = new THREE.Mesh(
    new THREE.SphereGeometry(0.275, 20, 20, 0, Math.PI * 2, 0, Math.PI * 0.62),
    hairMaterial,
  );
  hair.position.y = 0.3;
  headPivot.add(hair);

  const eyeGeometry = new THREE.SphereGeometry(0.022, 8, 8);
  const leftEye = new THREE.Mesh(eyeGeometry, eyeMaterial);
  leftEye.position.set(-0.09, 0.27, 0.235);
  headPivot.add(leftEye);
  const rightEye = new THREE.Mesh(eyeGeometry, eyeMaterial);
  rightEye.position.set(0.09, 0.27, 0.235);
  headPivot.add(rightEye);

  root.add(headPivot);

  const armMaterials: ArmMaterials = { sleeve: jacketMaterial, skin: skinMaterial };
  const left = buildArm(-1, armMaterials);
  const right = buildArm(1, armMaterials);
  root.add(left.shoulder, right.shoulder);

  // Everything above was authored around a y=0 waist (legs hang below it,
  // reaching roughly y=-0.73 at the shoes) -- shift the whole figure up so
  // the feet rest near the ground plane instead of floating under it.
  root.position.y = 0.75;

  return {
    root,
    head: headPivot,
    leftShoulder: left.shoulder,
    leftElbow: left.elbow,
    rightShoulder: right.shoulder,
    rightElbow: right.elbow,
  };
}

export function applyRotations(puppet: Puppet, rotations: JointRotations): void {
  puppet.head.rotation.set(rotations.head.x, rotations.head.y, 0);
  puppet.leftShoulder.rotation.set(rotations.leftShoulder.x, 0, rotations.leftShoulder.z);
  puppet.leftElbow.rotation.set(rotations.leftElbow, 0, 0);
  puppet.rightShoulder.rotation.set(rotations.rightShoulder.x, 0, rotations.rightShoulder.z);
  puppet.rightElbow.rotation.set(rotations.rightElbow, 0, 0);
}
