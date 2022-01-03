from __future__ import annotations

import math
from time import time_ns

import virxrlru as rlru
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

from utils import *
from vec import Matrix3, Vector

COAST_ACC = 525
BRAKE_ACC = 3500
MIN_BOOST_TIME = 3 / 120
REACTION_TIME = 0.04

BRAKE_COAST_TRANSITION = -(0.45 * BRAKE_ACC + 0.55 * COAST_ACC)  # -1863.75
COASTING_THROTTLE_TRANSITION = -0.5 * COAST_ACC  # -262.5
MIN_WALL_SPEED = -0.5 * BRAKE_ACC  # -1750

MAX_TURN_RADIUS = 1. / 0.00088

NO_ADJUST_RADIANS = 0.001
MIN_ADJUST_RADIANS = 0.5


class Bot(BaseAgent):
    def initialize_agent(self):
        self.ready = False
        self.me = car_object(self.index)
        self.boost_accel = 991 + 2 / 3
        self.time = 0
        self.tick_times = []

        team = [1, -1][self.team]
        self.target = (
            (team * 800, team * 5120, 321.3875),
            (team * -800, team * 5120, 321.3875),
        )

        rlru.load_soccar()

    def get_output(self, packet: GameTickPacket):
        if not self.ready:
            field_info = self.get_field_info()
            self.boosts = tuple(boost_object(i, field_info.boost_pads[i].location, field_info.boost_pads[i].is_full_boost) for i in range(field_info.num_boosts))
            self.ready = True

        start = time_ns()

        set(map(lambda pad: pad.update(packet), self.boosts))

        self.me.update(packet)
        self.time = packet.game_info.seconds_elapsed

        if self.me.airborne:
            return SimpleControllerState(throttle=1)

        rlru.tick(packet)
        
        shot = rlru.get_shot_with_target(self.target[0], self.target[1], self.index, {})

        if not shot['found']:
            if self.me.boost < 60:
                boosts = tuple(boost for boost in self.boosts if boost.active and boost.large)

                # if there's at least one large and active boost
                if len(boosts) > 0:
                    # Get the closest boost
                    closest_boost = min(boosts, key=lambda boost: boost.location.dist(self.me.location))

                    # Goto the nearest boost
                    local_final_target = self.me.local_location(closest_boost.location)
                    angle = math.atan2(local_final_target.y, local_final_target.x)
                    # return SimpleControllerState()
                    return SimpleControllerState(throttle=1, steer=cap((35 * angle) ** 3 / 10, -1, 1))
            
            local_final_target = self.me.local_location(Vector(y=(-1, 1)[self.team] * 5120))
            angle = math.atan2(local_final_target.y, local_final_target.x)
            # return SimpleControllerState()
            return SimpleControllerState(throttle=1, steer=cap((35 * angle) ** 3 / 10, -1, 1))

        future_ball_location = Vector(*rlru.get_slice(shot['time'])['location'])
        eta = shot['time']

        self.draw_point(future_ball_location, self.renderer.purple())

        T = eta - self.time

        shot_info = rlru.get_data_for_shot_with_target(self.target[0], self.target[1], eta, self.index, {})

        if len(shot_info['path_samples']) > 2:
            self.renderer.draw_polyline_3d(tuple(Vector(sample[0], sample[1], 30) for sample in shot_info['path_samples']), self.renderer.lime())
        else:
            self.renderer.draw_line_3d(tuple(self.me.location), tuple(shot_info['final_target']), self.renderer.lime())

        final_target = Vector(*shot_info['final_target'])
        self.draw_point(final_target, self.renderer.red())
        distance_remaining = shot_info['distance_remaining']

        local_final_target = self.me.local_location(final_target.flatten())

        car_speed = self.me.orientation.forward.dot(self.me.velocity)
        controller = SimpleControllerState()

        angle = math.atan2(local_final_target.y, local_final_target.x)
        controller.steer = cap(3.4 * angle + 0.235 * self.me.angular_velocity.z, -1, 1)

        if T > 0:
            speed_required = min(distance_remaining / T, 2300)

            t = speed_required - car_speed
            acceleration = t / REACTION_TIME

            brake_coast_transition = BRAKE_COAST_TRANSITION
            coasting_throttle_transition = COASTING_THROTTLE_TRANSITION
            throttle_accel = throttle_acceleration(car_speed)
            throttle_boost_transition = 1 * throttle_accel + 0.5 * self.boost_accel

            if acceleration <= brake_coast_transition:
                controller.throttle = -1

            elif (
                brake_coast_transition < acceleration
                and acceleration < coasting_throttle_transition
            ):
                pass

            elif (
                coasting_throttle_transition <= acceleration
                and acceleration <= throttle_boost_transition
            ):
                controller.throttle = (
                    1
                    if throttle_accel == 0
                    else cap(acceleration / throttle_accel, 0.02, 1)
                )

            elif throttle_boost_transition < acceleration:
                controller.throttle = 1
                if t > 0 and controller.steer < 1:
                    controller.boost = True

        end = time_ns()
        self.tick_times.append(round((end - start) / 1_000_000, 3))
        while len(self.tick_times) > 120:
            del self.tick_times[0]
        self.renderer.draw_string_3d(tuple(self.me.location), 2, 2, f"Intercept time: {round(eta, 2)}\nAverage ms/t: {round(sum(self.tick_times) / len(self.tick_times), 3)}", self.renderer.team_color(alt_color=True))

        # return SimpleControllerState()
        return controller

    def draw_point(self, point: Vector, color):
        self.renderer.draw_line_3d(
            (point - Vector(z=100)).to_vector3(),
            (point + Vector(z=100)).to_vector3(),
            color,
        )

    def draw_line(self, p1: Vector, p2: Vector, color):
        self.renderer.draw_line_3d(p1.to_vector3(), p2.to_vector3(), color)


class hitbox_object:
    def __init__(self, length=0, width=0, height=0):
        self.length = length
        self.width = width
        self.height = height
        self.offset = Vector()

    def __getitem__(self, index):
        return (self.length, self.width, self.height)[index]

    def from_car(self, car):
        self.length = car.hitbox.length
        self.width = car.hitbox.width
        self.height = car.hitbox.height
        self.offset = Vector.from_vector(car.hitbox_offset)


class boost_object:
    def __init__(self, index, location, large):
        self.index = index
        self.location = Vector.from_vector(location)
        self.active = True
        self.large = large

    def update(self, packet):
        self.active = packet.game_boosts[self.index].is_active

class car_object:
    # objects convert the gametickpacket in something a little friendlier to use
    # and are automatically updated by VirxERLU as the game runs
    def __init__(self, index):
        self.location = Vector()
        self.orientation = Matrix3(simple=True)
        self.velocity = Vector()
        self.angular_velocity = Vector()
        self.demolished = False
        self.airborne = False
        self.jumped = False
        self.doublejumped = False
        self.boost = 0
        self.index = index
        self.land_time = 0
        self.hitbox = hitbox_object()

    def local(self, value):
        return self.orientation.dot(value)

    def global_(self, value):
        return self.orientation.g_dot(value)

    def local_location(self, location):
        return self.local(location - self.location)

    def local_flatten(self, value):
        return self.global_(self.local(value).flatten())

    def get_raw(self):
        return {
            "location": tuple(self.location),
            "velocity": tuple(self.velocity),
            "angular_velocity": tuple(self.angular_velocity),
            "hitbox": tuple(self.hitbox),
            "hitbox_offset": tuple(self.hitbox.offset),
            "pitch": self.orientation.pitch,
            "yaw": self.orientation.yaw,
            "roll": self.orientation.roll,
            "boost": self.boost,
            "demolished": self.demolished,
            "airborne": self.airborne,
            "jumped": self.jumped,
            "doublejumped": self.doublejumped,
            "index": self.index,
        }

    def update(self, packet: GameTickPacket):
        car = packet.game_cars[self.index]
        car_phy = car.physics
        self.location = Vector.from_vector(car_phy.location)
        self.velocity = Vector.from_vector(car_phy.velocity)
        self.orientation = Matrix3.from_rotator(car_phy.rotation)
        self.raw_angular_velocity = Vector.from_vector(car_phy.angular_velocity)
        self.angular_velocity = self.orientation.dot(self.raw_angular_velocity)
        self.hitbox.from_car(car)
        self.demolished = car.is_demolished
        self.airborne = not car.has_wheel_contact
        self.jumped = car.jumped
        self.doublejumped = car.double_jumped
        self.boost = car.boost

        if self.airborne and car.has_wheel_contact:
            self.land_time = packet.game_info.seconds_elapsed
