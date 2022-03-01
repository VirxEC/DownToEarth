from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from rlbot.utils.structures.game_data_struct import Rotator, Vector3


# Vector supports 1D, 2D and 3D Vectors, as well as calculations between them
# Arithmetic with 1D and 2D lists/tuples aren't supported - just set the remaining values to 0 manually
# With this new setup, Vector is much faster because it's just a wrapper for numpy
class Vector:
    def __init__(self, x: float=0, y: float=0, z: float=0, np_arr: Optional[np.ndarray]=None):
        # this is a private property - this is so all other things treat this class like a list, and so should you!
        self._np = np.array([x, y, z]) if np_arr is None else np_arr

    def __getitem__(self, index: int):
        return self._np[index].item()

    def __setitem__(self, index: int, value: float):
        self._np[index] = value

    @property
    def x(self):
        return self._np[0].item()

    @x.setter
    def x(self, value: float):
        self._np[0] = value

    @property
    def y(self):
        return self._np[1].item()

    @y.setter
    def y(self, value: float):
        self._np[1] = value

    @property
    def z(self):
        return self._np[2].item()

    @z.setter
    def z(self, value: float):
        self._np[2] = value

    # self == value
    def __eq__(self, value: float|Vector):
        if isinstance(value, float) or isinstance(value, int):
            return self.magnitude() == value

        if hasattr(value, "_np"):
            value = value._np
        return (self._np == value).all()

    # len(self)
    def __len__(self):
        return 3  # this is a 3 dimensional vector, so we return 3

    # str(self)
    def __str__(self):
        # Vector's can be printed to console
        return f"[{self.x} {self.y} {self.z}]"

    # repr(self)
    def __repr__(self):
        return f"Vector(x={self.x}, y={self.y}, z={self.z})"

    # -self
    def __neg__(self):
        return Vector(np_arr=self._np * -1)

    # self + value
    def __add__(self, value):
        if hasattr(value, "_np"):
            value = value._np
        return Vector(np_arr=self._np+value)
    __radd__ = __add__

    # self - value
    def __sub__(self, value):
        if hasattr(value, "_np"):
            value = value._np
        return Vector(np_arr=self._np-value)

    def __rsub__(self, value):
        return -self + value

    # self * value
    def __mul__(self, value):
        if hasattr(value, "_np"):
            value = value._np
        return Vector(np_arr=self._np*value)
    __rmul__ = __mul__

    # self / value
    def __truediv__(self, value):
        if hasattr(value, "_np"):
            value = value._np
        return Vector(np_arr=self._np/value)

    def __rtruediv__(self, value):
        return self * (1 / value)

    # round(self)
    def __round__(self, decimals: int=0) -> Vector:
        # Rounds all of the values
        return Vector(np_arr=np.around(self._np, decimals=decimals))

    @staticmethod
    def from_vector(vec: Vector3) -> Vector:
        return Vector(vec.x, vec.y, vec.z)

    def magnitude(self) -> float:
        # Returns the length of the vector
        return np.linalg.norm(self._np).item()

    def _magnitude(self) -> np.float64:
        # Returns the length of the vector in a numpy float 64
        return np.linalg.norm(self._np)

    def dot(self, value: Vector|np.ndarray) -> float:
        # Returns the dot product of two vectors
        if hasattr(value, "_np"):
            value = value._np
        return self._np.dot(value).item()

    def cross(self, value: Vector|np.ndarray) -> Vector:
        # Returns the cross product of two vectors
        if hasattr(value, "_np"):
            value = value._np
        return Vector(np_arr=np.cross(self._np, value))

    def copy(self) -> Vector:
        # Returns a copy of the vector
        return Vector(*self._np)

    def normalize(self, return_magnitude: bool=False) -> Tuple[Vector, float] | Vector:
        # normalize() returns a Vector that shares the same direction but has a length of 1
        # normalize(True) can also be used if you'd like the length of this Vector (used for optimization)
        magnitude = self._magnitude()
        if magnitude != 0:
            norm_vec = Vector(np_arr=self._np / magnitude)
            if return_magnitude:
                return norm_vec, magnitude.item()
            return norm_vec
        if return_magnitude:
            return Vector(), 0
        return Vector()

    def _normalize(self) -> np.ndarray:
        # Normalizes a Vector and returns a numpy array
        magnitude = self._magnitude()
        if magnitude != 0:
            return self._np / magnitude
        return np.array((0, 0, 0))

    def flatten(self) -> Vector:
        # Sets Z (Vector[2]) to 0, making the Vector 2D
        return Vector(self._np[0], self._np[1])

    def angle2D(self, value: Vector) -> float:
        # Returns the 2D angle between this Vector and another Vector in radians
        return self.flatten().angle(value.flatten())

    def angle(self, value: Vector) -> float:
        # Returns the angle between this Vector and another Vector in radians
        dp = np.dot(self._normalize(), value._normalize()).item()
        return math.acos(-1 if dp < -1 else (1 if dp > 1 else dp))

    def rotate2D(self, angle: float) -> Vector:
        # Rotates this Vector by the given angle in radians
        # Note that this is only 2D, in the x and y axis
        return Vector((math.cos(angle)*self.x) - (math.sin(angle)*self.y), (math.sin(angle)*self.x) + (math.cos(angle)*self.y), self.z)

    def clamp2D(self, start: Vector, end: Vector) -> Vector:
        # Similar to integer clamping, Vector's clamp2D() forces the Vector's direction between a start and end Vector
        # Such that Start < Vector < End in terms of clockwise rotation
        # Note that this is only 2D, in the x and y axis
        s = self._normalize()
        right = np.dot(s, np.cross(end._np, (0, 0, -1))) < 0
        left = np.dot(s, np.cross(start._np, (0, 0, -1))) > 0
        if (right and left) if np.dot(end._np, np.cross(start._np, (0, 0, -1))) > 0 else (right or left):
            return self
        if np.dot(start._np, s) < np.dot(end._np, s):
            return end
        return start

    def clamp(self, start: Vector, end: Vector) -> Vector:
        # This extends clamp2D so it also clamps the vector's z
        s = self.clamp2D(start, end)

        if s.z < start.z:
            s = s.flatten().scale(1 - start.z)
            s.z = start.z
        elif s.z > end.z:
            s = s.flatten().scale(1 - end.z)
            s.z = end.z

        return s

    def dist(self, value: Vector|np.ndarray) -> float:
        # Distance between 2 vectors
        if hasattr(value, "_np"):
            value = value._np
        return np.linalg.norm(self._np - value).item()

    def flat_dist(self, value: Vector) -> float:
        # Distance between 2 vectors on a 2D plane
        return value.flatten().dist(self.flatten())

    def cap(self, low: float, high: float) -> Vector:
        # Caps all values in a Vector between 'low' and 'high'
        return Vector(*(max(min(item, high), low) for item in self._np))

    def midpoint(self, value: Vector|np.ndarray) -> Vector:
        # Midpoint of the 2 vectors
        if hasattr(value, "_np"):
            value = value._np
        return Vector(np_arr=(self._np + value) / 2)

    def scale(self, value: float) -> Vector:
        # Returns a vector that has the same direction but with a value as the magnitude
        return self.normalize() * value

        
class Matrix3:
    # The Matrix3's sole purpose is to convert roll, pitch, and yaw data from the gametickpacket into an orientation matrix
    # An orientation matrix contains 3 Vector's
    # Matrix3[0] is the "forward" direction of a given car
    # Matrix3[1] is the "right" direction of a given car
    # Matrix3[2] is the "up" direction of a given car
    # If you have a distance between the car and some object, ie ball.location - car.location,
    # you can convert that to local coordinates by dotting it with this matrix
    # ie: local_ball_location = Matrix3.dot(ball.location - car.location)
    # to convert from local coordinates back to global coordinates:
    # global_ball_location = Matrix3.g_dot(local_ball_location) + car_location
    def __init__(self, pitch: float=0, yaw: float=0, roll: float=0, simple: bool=False):
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll

        if simple:
            self._np = np.array(((0, 0, 0), (0, 0, 0), (0, 0, 0)))
            return

        CP = math.cos(self.pitch)
        SP = math.sin(self.pitch)
        CY = math.cos(self.yaw)
        SY = math.sin(self.yaw)
        CR = math.cos(self.roll)
        SR = math.sin(self.roll)
        # List of 3 vectors, each descriping the direction of an axis: Forward, Left, and Up
        self._np = np.array((
            (CP*CY, CP*SY, SP),
            (CY*SP*SR-CR*SY, SY*SP*SR+CR*CY, -CP*SR),
            (-CR*CY*SP-SR*SY, -CR*SY*SP+SR*CY, CP*CR)
        ))

    @property
    def rotation(self) -> Tuple[Vector, Vector, Vector]:
        return tuple(Vector(*item) for item in self._np)

    @property
    def forward(self) -> Vector:
        return Vector(*self._np[0])

    @property
    def right(self) -> Vector:
        return Vector(*self._np[1])

    @property
    def up(self) -> Vector:
        return Vector(*self._np[2])

    def __getitem__(self, key: int) -> Vector:
        return self.rotation[key]

    def __str__(self) -> str:
        return f"[{self.forward}\n {self.right}\n {self.up}]"

    @staticmethod
    def from_rotator(rotator: Rotator) -> Matrix3:
        return Matrix3(rotator.pitch, rotator.yaw, rotator.roll)

    @staticmethod
    def from_direction(direction: Vector, up: Vector) -> Matrix3:
        # once again, big thanks to Chip and his RLU
        # https://github.com/samuelpmish/RLUtilities/blob/develop/inc/linear_algebra/math.h

        mat = Matrix3(simple=True)
        forward = direction.normalize()
        up = forward.cross(up.cross(forward)).normalize()
        right = up.cross(forward).normalize()

        # generate the orientation matrix
        mat._np = np.array((tuple(forward), tuple(right), tuple(up)))
        mat.rotation = (forward, right, up)

        # generate the pitch/yaw/roll
        mat.pitch = math.atan2(mat.forward.z, Vector(mat.forward.x, mat.forward.y).magnitude())
        mat.yaw = math.atan2(mat.forward.y, mat.forward.x)
        mat.roll = math.atan2(-mat.right.z, mat.up.z)

        return mat

    def dot(self, vec: Vector) -> Vector:
        if hasattr(vec, "_np"):
            vec = vec._np
        return Vector(np_arr=self._np.dot(vec))

    def g_dot(self, vec: Vector) -> Vector:
        if hasattr(vec, "_np"):
            vec = vec._np
        return Vector(np_arr=self._np[0].dot(vec[0]) + self._np[1].dot(vec[1]) + self._np[2].dot(vec[2]))

    def det(self) -> Vector:
        return np.linalg.det(self._np).item()
