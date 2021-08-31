import math
from typing import Optional

from vec import Vector


def cap(x, low, high):
    return low if x < low else (high if x > high else x)


def turn_radius(v):
    # v is the magnitude of the velocity in the car's forward direction
    return 1.0 / curvature(v)


def curvature(v):
    # v is the magnitude of the velocity in the car's forward direction
    if 0 <= v < 500:
        return 0.0069 - 5.84e-6 * v

    if 500 <= v < 1000:
        return 0.00561 - 3.26e-6 * v

    if 1000 <= v < 1500:
        return 0.0043 - 1.95e-6 * v

    if 1500 <= v < 1750:
        return 0.003025 - 1.1e-6 * v

    if 1750 <= v < 2500:
        return 0.0018 - 4e-7 * v

    return 0


def sign(x: float) -> float:
    if x < 0:
        return -1

    if x > 0:
        return 1

    return 0


def throttle_acceleration(car_velocity_x: float) -> float:
    x = abs(car_velocity_x)
    if x >= 1410:
        return 0

    if x < 1400:
        return (-36 / 35) * x + 1600

    x -= 1400
    return -16 * x + 160


def radius_from_local_point(a: Vector) -> Optional[float]:
    # actually needs 3 points
    # point b is assumed to be at 0, 0
    c = Vector(-a.x, a.y)

    d1 = Vector(-a.y, a.x)
    d2 = Vector(c.y - a.y, a.x - c.x)

    k = d2.x * d1.y - d2.y * d1.x
    if -0.00001 < k < 0.00001:
        return None

    s1 = Vector(a.x / 2, a.y / 2)
    s2 = Vector((a.x + c.x) / 2, (a.y + c.y) / 2)
    l = d1.x * (s2.y - s1.y) - d1.y * (s2.x - s1.x)
    m = l / k
    center = Vector(s2.x + m * d2.x, s2.y + m * d2.y)

    dx = center.x - a.x
    dy = center.y - a.y
    radius = math.sqrt(dx * dx + dy * dy)

    return radius

    # try:
    #     b = a.flatten()
    #     return 1 / (2*b.y / a.dot(b))
    # except ZeroDivisionError:
    #     return None
