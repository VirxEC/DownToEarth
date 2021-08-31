from __future__ import annotations

import math

import numpy as np
from rlbot.utils.structures.game_data_struct import Vector3


class Vector:
    def __init__(self, x: float = 0, y: float = 0, z: float = 0, np_arr=None):
        self._np = np.array([x, y, z]) if np_arr is None else np_arr

    def __getitem__(self, index):
        return self._np[index].item()

    @property
    def x(self):
        return self._np[0].item()

    @x.setter
    def x(self, value):
        self._np[0] = value

    @property
    def y(self):
        return self._np[1].item()

    @y.setter
    def y(self, value):
        self._np[1] = value

    @property
    def z(self):
        return self._np[2].item()

    @z.setter
    def z(self, value):
        self._np[2] = value

    def __str__(self):
        return f"[{self.x} {self.y} {self.z}]"

    def __neg__(self):
        return Vector(np_arr=self._np * -1)

    def __add__(self, value):
        if hasattr(value, "_np"):
            value = value._np
        return Vector(np_arr=self._np + value)

    def __sub__(self, value):
        if hasattr(value, "_np"):
            value = value._np
        return Vector(np_arr=self._np - value)

    def __mul__(self, value):
        if hasattr(value, "_np"):
            value = value._np
        return Vector(np_arr=self._np * value)

    def __truediv__(self, value):
        if hasattr(value, "_np"):
            value = value._np
        return Vector(np_arr=self._np / value)

    def __round__(self, decimals=0) -> Vector:
        return Vector(np_arr=np.around(self._np, decimals=decimals))

    @staticmethod
    def from_vector(vec) -> Vector:
        return Vector(vec.x, vec.y, vec.z)

    def to_vector3(self) -> Vector3:
        return Vector3(self.x, self.y, self.z)

    def copy(self) -> Vector:
        return Vector(self.x, self.y, self.z)

    def magnitude(self) -> float:
        return np.linalg.norm(self._np).item()

    def _magnitude(self) -> np.float64:
        return np.linalg.norm(self._np)

    def dot(self, value: Vector) -> float:
        if hasattr(value, "_np"):
            value = value._np
        return self._np.dot(value).item()

    def cross(self, value: Vector) -> Vector:
        if hasattr(value, "_np"):
            value = value._np
        return Vector(np_arr=np.cross(self._np, value))

    def normalize(self) -> Vector:
        magnitude = self._magnitude()
        if magnitude != 0:
            return Vector(np_arr=self._np / magnitude)
        return Vector()

    def _normalize(self) -> np.ndarray:
        magnitude = self._magnitude()
        if magnitude != 0:
            return self._np / magnitude
        return np.array((0, 0, 0))

    def flatten(self) -> Vector:
        return Vector(self._np[0], self._np[1])

    def angle2D(self, value: Vector) -> float:
        dp = np.dot(self.flatten()._normalize(), value.flatten()._normalize()).item()
        return math.acos(-1 if dp < -1 else (1 if dp > 1 else dp))

    def angleTau2D(self, value: Vector) -> float:
        angle = math.atan2(value.y, value.x) - math.atan2(self.y, self.x)
        if angle < 0:
            angle += 2 * math.pi
        return angle

    def rotate2D(self, angle: float) -> Vector:
        # Rotates this Vector by the given angle in radians
        # Note that this is only 2D, in the x and y axis
        return Vector(
            (math.cos(angle) * self.x) - (math.sin(angle) * self.y),
            (math.sin(angle) * self.x) + (math.cos(angle) * self.y),
            self.z,
        )

    def clamp2D(self, start: Vector, end: Vector) -> Vector:
        s = self._normalize()
        right = np.dot(s, np.cross(end._np, (0, 0, -1))) < 0
        left = np.dot(s, np.cross(start._np, (0, 0, -1))) > 0
        if (
            (right and left)
            if np.dot(end._np, np.cross(start._np, (0, 0, -1))) > 0
            else (right or left)
        ):
            return self
        if np.dot(start._np, s) < np.dot(end._np, s):
            return end
        return start

    def dist(self, value: Vector) -> float:
        # Distance between 2 vectors
        if hasattr(value, "_np"):
            value = value._np
        return np.linalg.norm(self._np - value).item()

    def dist2D(self, value: Vector) -> float:
        return self.flatten().dist(value.flatten())

    def scale(self, value: float) -> Vector:
        # Returns a vector that has the same direction but with a value as the magnitude
        return self.normalize() * value


class Matrix3:
    def __init__(self, pitch=0, yaw=0, roll=0, simple=False):
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll

        if simple:
            self._np = np.array(((0, 0, 0), (0, 0, 0), (0, 0, 0)))
            self.rotation = (Vector(), Vector(), Vector())
            return

        CP = math.cos(self.pitch)
        SP = math.sin(self.pitch)
        CY = math.cos(self.yaw)
        SY = math.sin(self.yaw)
        CR = math.cos(self.roll)
        SR = math.sin(self.roll)
        # List of 3 vectors, each descriping the direction of an axis: Forward, Left, and Up
        self._np = np.array(
            (
                (CP * CY, CP * SY, SP),
                (CY * SP * SR - CR * SY, SY * SP * SR + CR * CY, -CP * SR),
                (-CR * CY * SP - SR * SY, -CR * SY * SP + SR * CY, CP * CR),
            )
        )

        self.rotation = tuple(Vector(*item) for item in self._np)

    @property
    def forward(self):
        return self.rotation[0]

    @property
    def right(self):
        return self.rotation[1]

    @property
    def up(self):
        return self.rotation[2]

    @staticmethod
    def from_rotator(rotator) -> Matrix3:
        return Matrix3(rotator.pitch, rotator.yaw, rotator.roll)

    def dot(self, vec: Vector) -> Vector:
        if hasattr(vec, "_np"):
            vec = vec._np
        return Vector(np_arr=self._np.dot(vec))

    def g_dot(self, vec: Vector) -> Vector:
        if hasattr(vec, "_np"):
            vec = vec._np
        return Vector(
            np_arr=self._np[0].dot(vec[0])
            + self._np[1].dot(vec[1])
            + self._np[2].dot(vec[2])
        )
