"""
Constraint Manager — Hierarchical Transform & Lock System
==========================================================

Architecture:
  - SnapConstraint: Records HOW two objects are aligned (face, edge, vertex)
  - ObjectGroup: Parent-child hierarchy with relative offsets
  - ConstraintManager: Central API for lock/unlock/propagation

Transform Propagation:
  When leader moves/rotates, children follow via:
    child_world_pos = group_pivot + R_delta @ (child_pos - group_pivot) + t_delta
    child_rotation += delta_rotation
"""

import numpy as np
import uuid
import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SnapConstraint:
    """Records how Object A relates to Object B after a snap."""
    constraint_id: str
    type: str              # "FACE_ALIGN", "EDGE_ALIGN", "VERTEX"
    obj_a_id: str
    obj_b_id: str
    face_a: str = ""       # "left", "right", "top", "bottom", "front", "back"
    face_b: str = ""
    offset_matrix: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))


@dataclass
class ObjectGroup:
    """A group of objects with hierarchical transform."""
    group_id: str
    leader_id: str                          # The "parent" object
    member_ids: list = field(default_factory=list)
    constraints: list = field(default_factory=list)  # SnapConstraints within this group
    
    # Stored relative transforms: member_id -> (relative_pos, relative_rot)
    # These are captured at lock time and used for propagation
    _relative_transforms: dict = field(default_factory=dict)


class ConstraintManager:
    """
    Central manager for all object groups and constraints.
    
    Usage:
        cm = ConstraintManager()
        cm.create_snap_lock(obj_a, obj_b, "right", "left")   # Auto-lock after face snap
        cm.create_manual_lock([obj_a, obj_b, obj_c])          # Manual grouping
        cm.propagate_movement(leader, delta_pos, delta_rot, all_objects)
        cm.break_lock(obj, all_objects)
    """
    
    def __init__(self):
        self.groups: dict[str, ObjectGroup] = {}  # group_id -> ObjectGroup
        self.constraints: list[SnapConstraint] = []
    
    def get_group_for_object(self, obj) -> Optional[ObjectGroup]:
        """Find which group an object belongs to, if any."""
        for group in self.groups.values():
            if obj.obj_id == group.leader_id or obj.obj_id in group.member_ids:
                return group
        return None
    
    def get_group_objects(self, group: ObjectGroup, all_objects: list) -> list:
        """Get all Object3D instances in a group."""
        ids = set([group.leader_id] + group.member_ids)
        return [o for o in all_objects if o.obj_id in ids]
    
    def get_group_leader(self, group: ObjectGroup, all_objects: list):
        """Get the leader Object3D of a group."""
        for o in all_objects:
            if o.obj_id == group.leader_id:
                return o
        return None
    
    def _compute_group_pivot(self, group: ObjectGroup, all_objects: list) -> np.ndarray:
        """Compute pivot point as center of all member bounding boxes."""
        members = self.get_group_objects(group, all_objects)
        if not members:
            return np.zeros(3, dtype=np.float32)
        positions = np.array([o.position for o in members])
        return positions.mean(axis=0)
    
    def _store_relative_transforms(self, group: ObjectGroup, all_objects: list):
        """
        Capture each member's position/rotation relative to the leader.
        Called once at lock time.
        """
        leader = self.get_group_leader(group, all_objects)
        if leader is None:
            return
        
        group._relative_transforms = {}
        members = self.get_group_objects(group, all_objects)
        
        for obj in members:
            if obj.obj_id == group.leader_id:
                continue
            # Store relative position and rotation
            rel_pos = obj.position - leader.position
            rel_rot = obj.rotation - leader.rotation
            group._relative_transforms[obj.obj_id] = (rel_pos.copy(), rel_rot.copy())
    
    # ──────────────────────────────────────────────────────────
    # LOCK / UNLOCK API
    # ──────────────────────────────────────────────────────────
    
    def create_snap_lock(self, obj_a, obj_b, face_a: str = "", face_b: str = "",
                         all_objects: list = None) -> str:
        """
        Create a lock between two objects after a snap event.
        If either object is already in a group, merge into that group.
        Returns the group_id.
        """
        group_a = self.get_group_for_object(obj_a)
        group_b = self.get_group_for_object(obj_b)
        
        # Record the constraint
        constraint = SnapConstraint(
            constraint_id=str(uuid.uuid4())[:8],
            type="FACE_ALIGN" if face_a else "VERTEX",
            obj_a_id=obj_a.obj_id,
            obj_b_id=obj_b.obj_id,
            face_a=face_a,
            face_b=face_b,
        )
        self.constraints.append(constraint)
        
        if group_a and group_b:
            # Merge group_b into group_a
            if group_a.group_id != group_b.group_id:
                for mid in group_b.member_ids:
                    if mid not in group_a.member_ids and mid != group_a.leader_id:
                        group_a.member_ids.append(mid)
                if group_b.leader_id not in group_a.member_ids and group_b.leader_id != group_a.leader_id:
                    group_a.member_ids.append(group_b.leader_id)
                group_a.constraints.extend(group_b.constraints)
                group_a.constraints.append(constraint)
                # Remove old group
                del self.groups[group_b.group_id]
            else:
                group_a.constraints.append(constraint)
            
            if all_objects:
                self._store_relative_transforms(group_a, all_objects)
            return group_a.group_id
            
        elif group_a:
            # Add obj_b to group_a
            if obj_b.obj_id not in group_a.member_ids:
                group_a.member_ids.append(obj_b.obj_id)
            group_a.constraints.append(constraint)
            if all_objects:
                self._store_relative_transforms(group_a, all_objects)
            return group_a.group_id
            
        elif group_b:
            # Add obj_a to group_b
            if obj_a.obj_id not in group_b.member_ids:
                group_b.member_ids.append(obj_a.obj_id)
            group_b.constraints.append(constraint)
            if all_objects:
                self._store_relative_transforms(group_b, all_objects)
            return group_b.group_id
            
        else:
            # Create new group. obj_b (the stationary one) is leader.
            gid = str(uuid.uuid4())[:8]
            group = ObjectGroup(
                group_id=gid,
                leader_id=obj_b.obj_id,
                member_ids=[obj_a.obj_id],
                constraints=[constraint],
            )
            self.groups[gid] = group
            if all_objects:
                self._store_relative_transforms(group, all_objects)
            return gid
    
    def create_manual_lock(self, objects: list, all_objects: list = None) -> str:
        """
        Manually group a list of objects.
        First object becomes the leader.
        """
        if len(objects) < 2:
            return ""
        
        # Check if any are already grouped — merge into that group
        existing_group = None
        for obj in objects:
            g = self.get_group_for_object(obj)
            if g:
                existing_group = g
                break
        
        if existing_group:
            for obj in objects:
                if obj.obj_id != existing_group.leader_id and obj.obj_id not in existing_group.member_ids:
                    existing_group.member_ids.append(obj.obj_id)
            if all_objects:
                self._store_relative_transforms(existing_group, all_objects)
            return existing_group.group_id
        
        gid = str(uuid.uuid4())[:8]
        leader = objects[0]
        members = [o.obj_id for o in objects[1:]]
        
        group = ObjectGroup(
            group_id=gid,
            leader_id=leader.obj_id,
            member_ids=members,
        )
        self.groups[gid] = group
        if all_objects:
            self._store_relative_transforms(group, all_objects)
        
        print(f"🔒 Manual Lock: Group {gid} ({len(objects)} objects)")
        return gid
    
    def break_lock(self, obj, all_objects: list = None):
        """Remove an object from its group. If only 1 remains, dissolve group."""
        group = self.get_group_for_object(obj)
        if not group:
            return
        
        if obj.obj_id == group.leader_id:
            # Removing leader — promote first member
            if group.member_ids:
                group.leader_id = group.member_ids.pop(0)
            else:
                # Dissolve
                del self.groups[group.group_id]
                print(f"🔓 Group {group.group_id} dissolved")
                return
        else:
            group.member_ids.remove(obj.obj_id)
        
        # Remove from relative transforms
        group._relative_transforms.pop(obj.obj_id, None)
        
        # If only leader remains, dissolve
        if not group.member_ids:
            del self.groups[group.group_id]
            print(f"🔓 Group {group.group_id} dissolved (only 1 object left)")
        elif all_objects:
            self._store_relative_transforms(group, all_objects)
        
        print(f"🔓 Broke lock for {obj.obj_id}")
    
    # ──────────────────────────────────────────────────────────
    # TRANSFORM PROPAGATION (MATRIX-BASED)
    # ──────────────────────────────────────────────────────────
    
    def _store_relative_transforms(self, group: ObjectGroup, all_objects: list):
        """
        Capture each member's transform relative to the leader using Matrix Math.
        M_child_local = inv(M_leader_world) @ M_child_world
        Called once at lock time.
        """
        leader = self.get_group_leader(group, all_objects)
        if leader is None:
            return
        
        group._relative_transforms = {}
        members = self.get_group_objects(group, all_objects)
        
        # Get Leader World Matrix
        # Note: get_model_matrix() returns np.array (4x4)
        M_leader = leader.get_model_matrix()
        try:
            M_leader_inv = np.linalg.inv(M_leader)
        except np.linalg.LinAlgError:
            print(f"⚠️ Singular matrix for leader {leader.name}, cannot lock.")
            return

        for obj in members:
            if obj.obj_id == group.leader_id:
                continue
            
            # M_child
            M_child = obj.get_model_matrix()
            
            # M_local = inv(M_parent) * M_child
            M_local = M_leader_inv @ M_child
            
            group._relative_transforms[obj.obj_id] = M_local
            
    def update_children_transforms(self, leader_obj, all_objects: list):
        """
        Re-compute all children's world transforms based on leader's new world transform.
        M_child_world = M_leader_world @ M_child_local
        This preserves relative position/rotation/scale perfectly.
        """
        group = self.get_group_for_object(leader_obj)
        if not group or group.leader_id != leader_obj.obj_id:
            return
        
        members = self.get_group_objects(group, all_objects)
        M_leader = leader_obj.get_model_matrix()
        
        from pyrr import Matrix44, Quaternion, Vector3
        
        for obj in members:
            if obj.obj_id == leader_obj.obj_id:
                continue
            
            # Retrieve stored local matrix
            M_local = group._relative_transforms.get(obj.obj_id)
            if M_local is None:
                continue
            
            # Calculate new world matrix
            # M_child_new = M_leader @ M_local
            M_new = M_leader @ M_local
            
            # Decompose M_new back to Pos, Rot, Scale
            # We use Pyrr for robust decomposition if possible, or manual extraction.
            # M_new is a 4x4 numpy array.
            
            # Translation is easy: M[3, 0..2] (if column-major? No, numpy/pyrr is row-major usually?
            # Wait, pyrr Matrix44 is Column-Major internally but .from_translation creates Row-Major friendly input?
            # Let's check Object3D.get_model_matrix:
            # t = Matrix44.from_translation(...) -> Row Major representation in memory?
            # Actually Pyrr is Column Major. But let's check translation index.
            # If M = T * R * S
            # Translation is M[3][0], M[3][1], M[3][2] in Row-Major (DirectX style) or M[0][3]... in Column-Major (OpenGL).
            # Pyrr is OpenGL style (Column Major).
            # However, when converted to numpy array via np.array(Matrix44), it might be transposed?
            # Let's verify standard Pyrr behavior.
            # actually Object3D uses `t = Matrix44.from_translation`
            # If I print(t), it looks like:
            # [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [x, y, z, 1]] (Row Major visual)
            # BUT efficient access.
            
            # Let's use Pyrr's decomposition if available, or just extract carefully.
            # Scale extraction is vector length of columns 0, 1, 2.
            
            # Convert numpy array back to Pyrr Matrix44 for convenience
            # (Assuming M_new is compatible)
            
            # Extract Position
            new_pos = M_new[3, :3] # Row 3 is translation in Pyrr/Numpy usually
            
            # Extract Scale
            sx = np.linalg.norm(M_new[0, :3])
            sy = np.linalg.norm(M_new[1, :3])
            sz = np.linalg.norm(M_new[2, :3])
            new_scale = np.array([sx, sy, sz], dtype=np.float32)
            
            # Extract Rotation
            # Remove scale from matrix to get pure rotation
            R_mat = M_new.copy()
            R_mat[0, :3] /= sx
            R_mat[1, :3] /= sy
            R_mat[2, :3] /= sz
            R_mat[3, :3] = 0
            R_mat[3, 3] = 1
            
            # Convert Rotation Matrix to Euler (yaw, pitch, roll)
            # Pyrr has quaternion support. Matrix -> Quat -> Euler is safer.
            q = Quaternion.from_matrix(Matrix44(R_mat))
            # Quat to euler? Pyrr doesn't have direct quat to euler?
            # It has Matrix44.from_eulers.
            # We need eulers to update obj.rotation (which is in DEGREES).
            
            # Manual Euler extraction from Rotation Matrix (ZYX order?)
            # Cy = sqrt(m00*m00 + m10*m10)
            # if Cy > 1e-6:
            #    x = atan2(m21, m22)
            #    y = atan2(-m20, Cy)
            #    z = atan2(m10, m00)
            # else:
            #    x = atan2(-m12, m11)
            #    y = atan2(-m20, Cy)
            #    z = 0
            
            # NOTE: Object3D.rotation is (Rx, Ry, Rz).
            # Object3D.get_model_matrix builds it as:
            # r = Matrix44.from_eulers((rx, ry, rz))
            # Pyrr eulers are (roll, pitch, yaw) -> (x, y, z)?
            
            # Let's blindly use a robust decompose function or library if possible.
            # Since we only strictly need it to "look right", we can try to use `pyrr.euler.index` if it exists.
            
            # Alternative: Construct a new Matrix44 and retrieve eulers? Not built-in.
            
            # Custom Extraction (ZYX convention for consistency with from_eulers)
            # m = R_mat
            # y (yaw/heading) = atan2(m[1,0], m[0,0]) ?
            
            # Let's stick to simple matrix multiplication for Position first.
            # Rotation is harder because we store Eulers.
            # If we don't extract Eulers correctly, we can't update obj.rotation.
            
            # Simplified approach: Decompose rotation logic
            # ry = atan2(-R[2,0], sqrt(R[0,0]**2 + R[1,0]**2))
            # rx = atan2(R[2,1], R[2,2])
            # rz = atan2(R[1,0], R[0,0])
            
            # Let's use a snippet for Euler extraction from 4x4 matrix
            # Assumes YXZ or ZYX order? Pyrr default is usually XYZ or ZYX.
            
            # Actually, `scipy.spatial.transform.Rotation` is robust for this.
            # Can I use scipy? Probably not installed.
            # Stick to numpy math.
            
            # Rotation Matrix to Euler Angles (XYZ)
            # thetaX = atan2(r21, r22)
            # thetaY = atan2(-r20, sqrt(r21^2 + r22^2))
            # thetaZ = atan2(r10, r00)
            
            m11, m12, m13 = R_mat[0,0], R_mat[0,1], R_mat[0,2]
            m21, m22, m23 = R_mat[1,0], R_mat[1,1], R_mat[1,2]
            m31, m32, m33 = R_mat[2,0], R_mat[2,1], R_mat[2,2]
            
            # Pitch (around X)
            rx = math.atan2(m23, m33)
            # Yaw (around Y)
            ry = math.asin(-m13) if -1 <= -m13 <= 1 else math.atan2(-m13, 0)
            # Roll (around Z)
            rz = math.atan2(m12, m11)
            
            # But wait, Gimbal lock?
            # Also need to convert radians to degrees.
            
            new_rx_deg = np.degrees(rx)
            new_ry_deg = np.degrees(ry)
            new_rz_deg = np.degrees(rz)
            
            # Better check:
            # sy = sqrt(m00*m00 + m10*m10)
            sy_val = math.sqrt(m11*m11 + m21*m21)
            if sy_val > 1e-6:
                 new_rx_deg = np.degrees(math.atan2(m32, m33))
                 new_ry_deg = np.degrees(math.atan2(-m31, sy_val))
                 new_rz_deg = np.degrees(math.atan2(m21, m11))
            else:
                 new_rx_deg = np.degrees(math.atan2(-m23, m22))
                 new_ry_deg = np.degrees(math.atan2(-m31, sy_val))
                 new_rz_deg = 0
            
            # Apply
            obj.position = new_pos
            obj.scale = new_scale
            obj.rotation = np.array([new_rx_deg, new_ry_deg, new_rz_deg], dtype=np.float32)
            
    # Legacy methods removed (propagate_movement, propagate_rotation, propagate_scale)
    # They are replaced by the unified update_children_transforms
    
    def get_all_group_member_ids(self, obj) -> set:
        """Get all object IDs in the same group as obj, including obj itself."""
        group = self.get_group_for_object(obj)
        if not group:
            return {obj.obj_id}
        return set([group.leader_id] + group.member_ids)
    
    def is_locked(self, obj) -> bool:
        """Check if an object is part of any group."""
        return self.get_group_for_object(obj) is not None
"""
Constraint Manager — Hierarchical Transform & Lock System
"""
