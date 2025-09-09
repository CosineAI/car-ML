import math
import random
from dataclasses import dataclass
from typing import Dict, Any


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class Terrain:
    def __init__(self, length: float = 2000.0, seed: int | None = None):
        self.length = float(length)
        self.rng = random.Random(seed)
        # Base params for smooth rolling hills
        # Amplitudes (world units) and frequencies (radians per world unit)
        self._A = [
            2.2 + self.rng.random() * 0.8,
            1.3 + self.rng.random() * 0.7,
            0.8 + self.rng.random() * 0.5,
        ]
        self._F = [
            0.055 + self.rng.random() * 0.03,
            0.12 + self.rng.random() * 0.04,
            0.26 + self.rng.random() * 0.05,
        ]
        self._P = [self.rng.random() * math.tau for _ in range(3)]
        self._base = 2.0 + self.rng.random() * 0.4

    def height(self, x: float) -> float:
        x = clamp(x, 0.0, self.length)
        y = self._base
        for A, F, P in zip(self._A, self._F, self._P):
            y += A * math.sin(F * x + P)
        return y

    def slope(self, x: float) -> float:
        # dy/dx
        x = clamp(x, 0.0, self.length)
        g = 0.0
        for A, F, P in zip(self._A, self._F, self._P):
            g += A * F * math.cos(F * x + P)
        return g

    def export_profile(self, step: float = 0.5) -> Dict[str, Any]:
        n = int(self.length / step) + 1
        xs = [i * step for i in range(n)]
        ys = [self.height(x) for x in xs]
        return {"xs": xs, "ys": ys, "length": self.length}


@dataclass
class CarParams:
    r_back: float
    r_front: float
    wheelbase: float
    body_base_ratio: float
    body_height: float
    omega: float

    @staticmethod
    def create_random(rng: random.Random) -> "CarParams":
        return CarParams(
            r_back=rng.uniform(0.35, 1.0),
            r_front=rng.uniform(0.35, 1.0),
            wheelbase=rng.uniform(1.0, 3.0),
            body_base_ratio=rng.uniform(0.5, 1.1),  # fraction of wheelbase
            body_height=rng.uniform(0.3, 1.6),
            omega=rng.uniform(1.4, 2.8),
        )

    def mutated(self, rng: random.Random, scale: float = 0.15) -> "CarParams":
        def n(x, lo, hi, s=scale):
            return clamp(x + rng.gauss(0, s * (hi - lo)), lo, hi)

        return CarParams(
            r_back=n(self.r_back, 0.3, 1.2),
            r_front=n(self.r_front, 0.3, 1.2),
            wheelbase=n(self.wheelbase, 0.8, 3.5),
            body_base_ratio=n(self.body_base_ratio, 0.4, 1.3),
            body_height=n(self.body_height, 0.2, 2.0),
            omega=n(self.omega, 1.0, 3.2),
        )


@dataclass
class CarState:
    x_back: float
    time: float = 0.0
    stuck_time: float = 0.0
    done: bool = False

    @property
    def x_front(self) -> float:
        return self.x_back  # placeholder; will be set by simulator each frame

    def copy(self) -> "CarState":
        return CarState(
            x_back=self.x_back,
            time=self.time,
            stuck_time=self.stuck_time,
            done=self.done,
        )


class Simulator:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.terrain = Terrain(length=2000.0, seed=self.rng.randrange(1_000_000))
        self.best_params: CarParams | None = None
        self.best_distance: float = 0.0
        self.attempt: int = 0

        self._init_new_car(random_init=True)
        self._just_finished = False

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng.seed(seed)
        self.terrain = Terrain(length=2000.0, seed=self.rng.randrange(1_000_000))
        self.best_params = None
        self.best_distance = 0.0
        self.attempt = 0
        self._init_new_car(random_init=True)
        self._just_finished = False

    def _init_new_car(self, random_init: bool = False):
        if random_init or self.best_params is None:
            self.params = CarParams.create_random(self.rng)
        else:
            # Hill climbing with occasional exploration
            if self.rng.random() < 0.2:
                self.params = CarParams.create_random(self.rng)
            else:
                self.params = self.best_params.mutated(self.rng, scale=0.18)

        # Start just at x=0
        self.state = CarState(x_back=0.0)
        self._x_front = self.state.x_back + self.params.wheelbase
        self._y_back = self.terrain.height(self.state.x_back) + self.params.r_back
        self._y_front = self.terrain.height(self._x_front) + self.params.r_front
        self._distance = self._x_front

    def _step_once(self, dt: float = 0.05):
        if self.state.done:
            return

        # Kinematics on terrain
        L = self.params.wheelbase
        xb = self.state.x_back
        xf = xb + L

        # clamp to terrain
        xb = clamp(xb, 0.0, self.terrain.length)
        xf = clamp(xf, 0.0, self.terrain.length)

        yb_ground = self.terrain.height(xb)
        yf_ground = self.terrain.height(xf)
        yb = yb_ground + self.params.r_back
        yf = yf_ground + self.params.r_front

        gb = self.terrain.slope(xb)
        gf = self.terrain.slope(xf)

        # Effective forward speeds per wheel (units/sec)
        c_penalty = 0.55  # uphill penalty weight
        vb = max(0.0, self.params.omega * self.params.r_back - c_penalty * max(0.0, gb))
        vf = max(0.0, self.params.omega * self.params.r_front - c_penalty * max(0.0, gf))
        v = min(vb, vf)
        v = clamp(v, 0.0, 4.0)  # speed cap

        # Very steep slope threshold: traction loss
        slope_limit = 0.95  # ~54 deg
        if abs(math.atan(gb)) > slope_limit or abs(math.atan(gf)) > slope_limit:
            v = 0.0

        dx = v * dt
        if dx < 1e-4:
            self.state.stuck_time += dt
        else:
            self.state.stuck_time = 0.0

        xb_new = xb + dx
        xf_new = xb_new + L
        # clamp
        xb_new = clamp(xb_new, 0.0, self.terrain.length)
        xf_new = clamp(xf_new, 0.0, self.terrain.length)

        # Update state
        self.state.x_back = xb_new
        self._x_front = xf_new
        self._y_back = self.terrain.height(xb_new) + self.params.r_back
        self._y_front = self.terrain.height(xf_new) + self.params.r_front
        self._distance = self._x_front
        self.state.time += dt

        # Termination conditions
        time_limit = 60.0
        stuck_limit = 1.4
        near_end = self._x_front >= (self.terrain.length - 2.0)

        if self.state.stuck_time > stuck_limit or near_end or self.state.time > time_limit:
            self.state.done = True

    def _finalize_attempt(self):
        dist = float(self._distance)
        if dist > self.best_distance:
            self.best_distance = dist
            self.best_params = self.params
        self.attempt += 1
        self._init_new_car(random_init=(self.best_params is None))
        self._just_finished = True

    def _frame_dict(self) -> Dict[str, Any]:
        # Orientation
        L = self.params.wheelbase
        phi = math.atan2(self._y_front - self._y_back, L)

        # Camera tracks slightly ahead of car
        cam_x = max(0.0, (self.state.x_back + self._x_front) * 0.5 - 8.0)

        # Body base length in world units (fraction of wheelbase)
        body_base_len = clamp(self.params.body_base_ratio * L, 0.3 * L, 1.6 * L)

        return {
            "attempt": int(self.attempt + 1),
            "best_distance": float(self.best_distance),
            "current_distance": float(self._distance),
            "camera_x": float(cam_x),
            "phi": float(phi),
            "body_base_len": float(body_base_len),
            "body_height": float(self.params.body_height),
            "back_wheel": {
                "x": float(self.state.x_back),
                "y": float(self._y_back),
                "r": float(self.params.r_back),
            },
            "front_wheel": {
                "x": float(self._x_front),
                "y": float(self._y_front),
                "r": float(self.params.r_front),
            },
            "done": bool(self.state.done),
        }

    def next_frame(self, steps: int = 1) -> Dict[str, Any]:
        # If just finished in previous call, clear the flag and continue with new car
        if self._just_finished:
            self._just_finished = False

        if steps < 0:
            steps = 0
        for _ in range(int(steps)):
            if self.state.done:
                # finalize and start new
                self._finalize_attempt()
                break
            self._step_once()

        # After stepping, if done, finalize but allow JS to see a 'done' frame first on next call
        if self.state.done:
            frame = self._frame_dict()
            # Mark done true for this frame, but do not start new car until the next call
            self._finalize_attempt()
            frame["done"] = True
            return frame

        return self._frame_dict()

    def export_terrain_profile(self) -> Dict[str, Any]:
        return self.terrain.export_profile(step=0.5)