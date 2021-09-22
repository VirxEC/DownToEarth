import math
import re
from queue import Queue
from threading import Thread
from typing import List

import requests
from rlbot.utils.structures.ball_prediction_struct import BallPrediction
from rlbot.utils.structures.game_data_struct import GameTickPacket, Vector3

from vec import Vector


def cap(x, min_, max_):
    return min_ if x < min_ else (max_ if x > max_ else x)


class CarHeuristic:
    NAMES = (
        "may_ground_shot",
        "may_jump_shot",
        "may_double_jump_shot",
        "may_aerial"
    )

    def __init__(self):
        self.profile = [0.9, 0.9, 0.9, 0.9]

    def __str__(self):
        return str(self.profile)

    __repr__ = __str__

    def __len__(self) -> int:
        return len(self.profile)

    def __getitem__(self, index) -> float:
        return self.profile[index]

    def __setitem__(self, index, value):
        self.profile[index] = value


class PacketHeuristics:
    def __init__(self, threshold: float=0.8, gain: float=0.21, loss: float=0.005, unpause_delay: float=1.5, ignore_indexes: List[int]=[]):
        self.cars = {}
        self.car_tracker = {}
        self.init = False

        self.threshold = threshold
        self.gain = gain
        self.loss = loss

        self.ignore_indexes = ignore_indexes
        self.start_time = -1
        self.time = 0
        self.last_ball_touch_time = -1
        self.unpause_delay = unpause_delay
        self.last_pause_time = -1
        self.team_count = [0, 0]

        field_half_width = 4096
        field_third_width = field_half_width / 3

        field_half_length = 5120
        field_third_length = field_half_length / 3
        
        # the following comments are from the perspective of this diagram -> https://github.com/RLBot/RLBot/wiki/Useful-Game-Values
        self.zones = (
            (
                None,
                Zone2D(Vector(-field_third_width, 5120, 20), Vector(field_third_width, 6000, 20)),  # orange net
                None,
            ),
            (
                Zone2D(Vector(field_third_width, field_third_length, 20), Vector(field_half_width, field_half_length, 20)),  # orange field left
                Zone2D(Vector(-field_third_width, field_third_length, 20), Vector(field_third_width, field_half_length, 20)),  # orange field
                Zone2D(Vector(-field_half_width, field_third_length, 20), Vector(-field_third_width, field_half_length, 20)),  # orange field right
            ),
            (
                Zone2D(Vector(field_third_width, -field_third_length, 20), Vector(field_half_width, field_third_length, 20)),  # mid field left
                Zone2D(Vector(-field_third_width, -field_third_length, 20), Vector(field_third_width, field_third_length, 20)),  # mid field
                Zone2D(Vector(-field_half_width, -field_third_length, 20), Vector(-field_third_width, field_third_length, 20)),  # mid field right
            ),
            (
                Zone2D(Vector(field_third_width, -field_half_length, 20), Vector(field_half_width, -field_third_length, 20)), # blue field left
                Zone2D(Vector(-field_third_width, -field_half_length, 20), Vector(field_third_width, -field_third_length, 20)),  # blue field
                Zone2D(Vector(-field_half_width, -field_half_length, 20), Vector(-field_third_width, -field_third_length, 20)),  # blue field right
            ),
            (
                None,
                Zone2D(Vector(-field_third_width, -6000, 20), Vector(field_third_width, -5120, 20)),  # blue net
                None,
            )
        )

        self.field_dimensions = [len(self.zones), len(self.zones[0])]

    def add_tick(self, packet: GameTickPacket, ball_prediction_struct: BallPrediction) -> bool:
        time = packet.game_info.seconds_elapsed
        delta_time = time - self.time
        self.time = time

        if self.start_time == -1:
            self.start_time = self.time

        if not packet.game_info.is_round_active or packet.game_info.is_kickoff_pause:
            self.last_pause_time = self.time
            if not packet.game_info.is_round_active:
                return False

        team_count = [0, 0]

        for i in range(packet.num_cars):
            team_count[packet.game_cars[i].team] += 1

        self.team_count = team_count

        loss = self.loss * delta_time

        latest_touch = packet.game_ball.latest_touch
        handled_touch = latest_touch.time_seconds != self.last_ball_touch_time
        self.last_ball_touch_time = latest_touch.time_seconds
        
        ball_zone_id = self.get_zone_id(packet.game_ball.physics.location)

        future_zone_ids = set()
        future_ball_zone_ids = []
        for slice_ in ball_prediction_struct.slices[::15]:
            ball_location = slice_.physics.location
            future_zone_id = self.get_zone_id(ball_location)

            if future_zone_id not in future_zone_ids:
                future_zone_ids.add(future_zone_id)
                future_ball_zone_ids.append((
                    future_zone_id,
                    ball_location
                ))

        for i in range(packet.num_cars):
            if i in self.ignore_indexes:
                continue

            car = packet.game_cars[i]
            if car.is_demolished:
                continue

            if car.name not in self.car_tracker:
                self.car_tracker[car.name] = {
                    "last_wheel_contact": {
                        "time": -1,
                        "up": Vector(),
                        "location": Vector()
                    },
                    "zone_id": -1,
                    "friends": -1,
                    "foes": -1
                }

            if car.name not in self.cars:
                self.cars[car.name] = {}

            friends = self.car_tracker[car.name]['friends'] = self.get_friend_count(car.team)
            foes = self.car_tracker[car.name]['foes'] = self.get_foe_count(car.team)

            if friends not in self.cars[car.name]:
                self.cars[car.name][friends] = {}

            if foes not in self.cars[car.name][friends]:
                self.cars[car.name][friends][foes] = {}

            zone_id = self.car_tracker[car.name]['zone_id'] = self.get_zone_id(car.physics.location)

            if zone_id is None:
                print(f"WARNING: zone_id for {car.name} was None")
                continue

            if len(self.cars[car.name][friends][foes]) == 0:
                self.cars[car.name][friends][foes] = {i: CarHeuristic() for i in range(self.field_dimensions[0] * self.field_dimensions[1])}
            elif zone_id not in self.cars[car.name][friends][foes]:
                self.cars[car.name][friends][foes][zone_id] = CarHeuristic()

            if car.has_wheel_contact:
                self.car_tracker[car.name]['last_wheel_contact']['time'] = self.time
                self.car_tracker[car.name]['last_wheel_contact']['location'] = Vector.from_vector(car.physics.location)
                CP = math.cos(car.physics.rotation.pitch)
                SP = math.sin(car.physics.rotation.pitch)
                CY = math.cos(car.physics.rotation.yaw)
                SY = math.sin(car.physics.rotation.yaw)
                CR = math.cos(car.physics.rotation.roll)
                SR = math.sin(car.physics.rotation.roll)
                self.car_tracker[car.name]['last_wheel_contact']['up'] = Vector(-CR*CY*SP-SR*SY, -CR*SY*SP+SR*CY, CP*CR)

            if packet.game_info.is_kickoff_pause or self.time - self.last_pause_time < self.unpause_delay:
                continue

            # Ball heuristic
            surrounding_zone_ids = self.get_surrounding_zone_ids(zone_id)

            if ball_zone_id in surrounding_zone_ids:
                car_loss = loss / (friends + 1)

                ball_sections = set()
                for future_ball_zone_id, location in future_ball_zone_ids:
                    ball_section = self.get_ball_section(location, car.name)
                    if ball_section not in ball_sections:
                        ball_sections.add(ball_section)

                    self.cars[car.name][friends][foes][future_ball_zone_id][ball_section] = max(self.cars[car.name][friends][foes][future_ball_zone_id][ball_section] - car_loss, 0)

                all_zones = future_zone_ids.copy()
                for zone_id_ in surrounding_zone_ids:
                    if zone_id_ not in all_zones:
                        all_zones.add(zone_id_)
                        self.cars[car.name][friends][foes][zone_id_][ball_section] = max(self.cars[car.name][friends][foes][zone_id_][ball_section] - car_loss, 0)

                if not handled_touch and latest_touch.player_index == i and latest_touch.time_seconds > self.start_time:
                    time_airborne = self.time - self.car_tracker[car.name]['last_wheel_contact']['time']
                    divisors = [
                        car.has_wheel_contact,
                        1 in ball_sections and car.jumped,
                        {2, 3} & ball_sections and car.jumped and car.double_jumped,
                        {2, 3} & ball_sections and (time_airborne > 0.75 or not car.jumped),
                        True  # We're just going to ignore this touch
                    ]
                    ball_touch_section = divisors.index(True)
                    if ball_touch_section != 4:
                        self.cars[car.name][friends][foes][zone_id][ball_touch_section] = min(self.cars[car.name][friends][foes][zone_id][ball_touch_section] + self.gain + car_loss, 1)
            
        return True
    
    def get_ball_section(self, ball_location: Vector3, car_name: str) -> int:
        location = Vector.from_vector(ball_location) - self.car_tracker[car_name]['last_wheel_contact']['location']
        
        dbz = self.car_tracker[car_name]['last_wheel_contact']['up'].dot(location)
        divisors = [
            dbz <= 126.75,
            dbz <= 312.75,
            dbz <= 542.75,
            True
        ]

        return divisors.index(True)

    def get_friend_count(self, car_team: int):
        return self.team_count[car_team] - 1
    
    def get_foe_count(self, car_team: int):
        return self.team_count[not car_team]

    def get_zone_id(self, location: Vector3):
        for id_0, zones in enumerate(self.zones):
            for id_1, zone in enumerate(zones):
                if zone != None and zone.intersect_point(location):
                    return id_0 * 3 + id_1

    def get_surrounding_zone_ids(self, zone_id: int) -> List[int]:
        zone_id_0 = zone_id // 3
        zone_id_1 = zone_id % 3
        zone_ids = []

        for id_0 in range(-1, 2, 1):
            id_0 += zone_id_0
            
            if -1 < id_0 < self.field_dimensions[0]:
                for id_1 in range(-1, 2, 1):
                    id_1 += zone_id_1

                    if -1 < id_1 < self.field_dimensions[1] and self.zones[id_0][id_1] is not None:
                        zone_ids.append(id_0 * 3 + id_1)

        return zone_ids

    def get_car(self, car_name: str) -> CarHeuristic:
        if car_name not in self.cars or car_name not in self.car_tracker:
            return None

        return self.cars[car_name][self.car_tracker[car_name]['friends']][self.car_tracker[car_name]['foes']][self.car_tracker[car_name]['zone_id']]

    def predict_car(self, car: CarHeuristic) -> dict:
        return {car.NAMES[i]: car[i] > self.threshold for i in range(len(car))}


class Zone2D:
    def __init__(self, min_: Vector, max_: Vector):
        self.min = min_
        self.max = max_

    def intersect_sphere(self, l: Vector, r: float) -> bool:
        nearest = Vector(
            cap(l.x, self.min.x, self.max.x),
            cap(l.y, self.min.y, self.max.y),
            cap(l.z, self.min.z, self.max.z)
        )

        return (l - nearest).magnitude() <= r

    def intersect_point(self, b: Vector3) -> bool:
        return self.min.x <= b.x and self.max.x >= b.x and self.min.y <= b.y and self.max.y >= b.y


class ProfileHandler(Thread):
    def __init__(self):
        super().__init__()
        self.packets = Queue()
        self.eph = PacketHeuristics()

    def stop(self):
        for name, friends_data in self.eph.cars.items():
            for num_friends, foes_data in friends_data.items():
                for num_foes, data in foes_data.items():
                    while 1:
                        try:
                            requests.post("https://ml-online-collab.herokuapp.com/set_bot", data={"name": name, "num_foes": num_foes, "num_friends": num_friends, "data": data})
                        except Exception as e:
                            print(e)

    def run(self):
        profiles = {}

        while 1:
            try:
                packet = self.packets.get()

                names = (re.split(r' \(\d+\)$', packet.game_cars[i].name)[0] for i in packet.num_cars if i != self.index)
                for name in names:
                    if name not in profiles:
                        while 1:
                            try:
                                r = requests.post("https://ml-online-collab.herokuapp.com/get_bot", data={"name": name})
                                self.eph.cars[name] = r.json()
                                break
                            except Exception as e:
                                print(e)
                
                self.eph.add_tick(packet)
            except Exception:
                continue

