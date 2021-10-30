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
    try:
        b = a.flatten()
        return abs(1 / (2*b.y / a.dot(b)))
    except ZeroDivisionError:
        return None
