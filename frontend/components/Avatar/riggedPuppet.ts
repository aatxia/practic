/**
 * Loads and drives the real rigged/skinned/textured "Cesium Man" glTF
 * model (CC BY 4.0, © 2017 Cesium -- github.com/KhronosGroup/
 * glTF-Sample-Assets/tree/main/Models/CesiumMan) as an alternative to the
 * procedural primitive puppet (puppet.ts). Avatar.tsx builds the
 * procedural puppet first (always available, zero network dependency) and
 * swaps in this one if/when it finishes loading -- so a slow or failed
 * fetch never leaves the avatar blank or broken, only a bit less polished.
 *
 * This rig's bones were authored with their own local axis conventions,
 * unrelated to puppet.ts's hand-built pivots -- see riggedPoses.ts's
 * module docstring for how the rotation values there were actually
 * determined (empirically, via world-space coordinates, not guessed).
 */
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import type { JointRotations } from "./poses";

const MODEL_URL = "/avatar/CesiumMan.glb";

/** This rig's actual bone names (from the glTF's node list), not a
 * standard/portable skeleton naming convention -- specific to this one
 * downloaded asset. */
const BONE_NAMES = {
  leftShoulder: "Skeleton_arm_joint_L__4_",
  leftElbow: "Skeleton_arm_joint_L__3_",
  rightShoulder: "Skeleton_arm_joint_R",
  rightElbow: "Skeleton_arm_joint_R__2_",
  neck: "Skeleton_neck_joint_2",
} as const;

type BoneKey = keyof typeof BONE_NAMES;

export interface RiggedPuppet {
  root: THREE.Object3D;
  bones: Record<BoneKey, THREE.Bone>;
  bindQuaternions: Map<THREE.Bone, THREE.Quaternion>;
}

export class RiggedModelError extends Error {}

export async function loadRiggedPuppet(): Promise<RiggedPuppet> {
  const loader = new GLTFLoader();
  const gltf = await loader.loadAsync(MODEL_URL);
  const root = gltf.scene;

  const found: Partial<Record<BoneKey, THREE.Bone>> = {};
  const bindQuaternions = new Map<THREE.Bone, THREE.Quaternion>();
  const nameToKey = new Map<string, BoneKey>(
    (Object.entries(BONE_NAMES) as [BoneKey, string][]).map(([key, name]) => [name, key]),
  );

  root.traverse((obj) => {
    const bone = obj as THREE.Bone;
    if (!bone.isBone) {
      return;
    }
    bindQuaternions.set(bone, bone.quaternion.clone());
    const key = nameToKey.get(bone.name);
    if (key) {
      found[key] = bone;
    }
  });

  const missing = (Object.keys(BONE_NAMES) as BoneKey[]).filter((key) => !found[key]);
  if (missing.length > 0) {
    throw new RiggedModelError(`Rigged model is missing expected bone(s): ${missing.join(", ")}`);
  }

  return { root, bones: found as Record<BoneKey, THREE.Bone>, bindQuaternions };
}

const WORLD_X = new THREE.Vector3(1, 0, 0);
const WORLD_Y = new THREE.Vector3(0, 1, 0);
const WORLD_Z = new THREE.Vector3(0, 0, 1);

/**
 * Rotates `bone` by the given angles around WORLD-space X/Y/Z axes
 * (composed X, then Y, then Z), relative to its bind-pose quaternion --
 * correct regardless of the bone's own authored local axis convention,
 * because the desired rotation is expressed in world space and converted
 * to the bone's local space via its parent's ACTUAL current world
 * orientation, never an assumption about what "local X" means for this
 * particular bone.
 */
export function applyWorldSwing(
  bone: THREE.Bone,
  bindQuaternion: THREE.Quaternion,
  angleX: number,
  angleY: number,
  angleZ: number,
): void {
  if (!bone.parent) {
    return;
  }
  const parentWorldQuat = new THREE.Quaternion();
  bone.parent.getWorldQuaternion(parentWorldQuat);

  const deltaWorld = new THREE.Quaternion()
    .setFromAxisAngle(WORLD_X, angleX)
    .multiply(new THREE.Quaternion().setFromAxisAngle(WORLD_Y, angleY))
    .multiply(new THREE.Quaternion().setFromAxisAngle(WORLD_Z, angleZ));

  const deltaLocal = parentWorldQuat.clone().invert().multiply(deltaWorld).multiply(parentWorldQuat);
  bone.quaternion.copy(deltaLocal.multiply(bindQuaternion));
}

function swing(puppet: RiggedPuppet, key: BoneKey, angleX: number, angleY: number, angleZ: number): void {
  const bone = puppet.bones[key];
  const bind = puppet.bindQuaternions.get(bone);
  if (!bind) {
    return;
  }
  applyWorldSwing(bone, bind, angleX, angleY, angleZ);
}

/** Order matters for a correct skeletal pose: shoulders must be updated
 * (and their world matrices refreshed) before elbows, since an elbow's
 * world-space swing is expressed relative to its shoulder's CURRENT world
 * orientation. */
export function applyRiggedRotations(puppet: RiggedPuppet, rotations: JointRotations): void {
  swing(puppet, "leftShoulder", rotations.leftShoulder.x, 0, rotations.leftShoulder.z);
  puppet.root.updateMatrixWorld(true);
  swing(puppet, "leftElbow", rotations.leftElbow, 0, 0);

  swing(puppet, "rightShoulder", rotations.rightShoulder.x, 0, rotations.rightShoulder.z);
  puppet.root.updateMatrixWorld(true);
  swing(puppet, "rightElbow", rotations.rightElbow, 0, 0);

  swing(puppet, "neck", rotations.head.x, rotations.head.y, 0);
  puppet.root.updateMatrixWorld(true);
}
