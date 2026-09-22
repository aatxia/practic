import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { applyWorldSwing } from "./riggedPuppet";

function quatsClose(a: THREE.Quaternion, b: THREE.Quaternion, epsilon = 1e-5): boolean {
  // A quaternion and its negation represent the same rotation.
  const same = Math.abs(a.x - b.x) < epsilon && Math.abs(a.y - b.y) < epsilon &&
    Math.abs(a.z - b.z) < epsilon && Math.abs(a.w - b.w) < epsilon;
  const negated = Math.abs(a.x + b.x) < epsilon && Math.abs(a.y + b.y) < epsilon &&
    Math.abs(a.z + b.z) < epsilon && Math.abs(a.w + b.w) < epsilon;
  return same || negated;
}

describe("applyWorldSwing", () => {
  it("with an unrotated parent and identity bind, produces exactly the requested world rotation", () => {
    const parent = new THREE.Object3D();
    const bone = new THREE.Bone();
    parent.add(bone);
    parent.updateMatrixWorld(true);

    const bind = new THREE.Quaternion(); // identity
    applyWorldSwing(bone, bind, Math.PI / 2, 0, 0);
    parent.updateMatrixWorld(true);

    const worldQuat = new THREE.Quaternion();
    bone.getWorldQuaternion(worldQuat);

    const expected = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 2);
    expect(quatsClose(worldQuat, expected)).toBe(true);
  });

  it("a zero-angle swing restores exactly the bind pose, even with a rotated parent", () => {
    const parent = new THREE.Object3D();
    parent.quaternion.setFromAxisAngle(new THREE.Vector3(0, 1, 0), 1.234);
    const bone = new THREE.Bone();
    parent.add(bone);
    parent.updateMatrixWorld(true);

    const bind = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), 0.4);
    applyWorldSwing(bone, bind, 0, 0, 0);

    expect(quatsClose(bone.quaternion, bind)).toBe(true);
  });

  it("rotating around world X moves a point on world Y toward world Z, regardless of parent orientation", () => {
    const parent = new THREE.Object3D();
    parent.quaternion.setFromAxisAngle(new THREE.Vector3(0, 1, 0), 0.9); // arbitrary parent rotation
    const bone = new THREE.Bone();
    parent.add(bone);
    parent.updateMatrixWorld(true);

    // Bind: bone's local +Y axis points along world +Y (compensating for
    // the parent's rotation), so a swing around world X has an
    // unambiguous, easy-to-check effect on it.
    const parentWorldQuat = new THREE.Quaternion();
    parent.getWorldQuaternion(parentWorldQuat);
    const bind = parentWorldQuat.clone().invert();
    bone.quaternion.copy(bind);
    parent.updateMatrixWorld(true);

    applyWorldSwing(bone, bind, Math.PI / 2, 0, 0);
    parent.updateMatrixWorld(true);

    const worldQuat = new THREE.Quaternion();
    bone.getWorldQuaternion(worldQuat);
    const localUp = new THREE.Vector3(0, 1, 0).applyQuaternion(worldQuat);

    expect(localUp.y).toBeCloseTo(0, 5);
    expect(localUp.z).toBeCloseTo(1, 5);
  });

  it("does nothing to a bone with no parent, instead of throwing", () => {
    const bone = new THREE.Bone();
    const bind = new THREE.Quaternion();
    expect(() => applyWorldSwing(bone, bind, 1, 1, 1)).not.toThrow();
  });
});
