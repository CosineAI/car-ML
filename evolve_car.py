#!/usr/bin/env python3
"""
A simple evolutionary "machine learning" program that evolves a 2D car to traverse random terrain.

- Generates a random 2D terrain (continuous function).
- Defines cars with parameters:
    - rear wheel radius (r1)
    - front wheel radius (r2)
    - chassis (axle-to-axle) length (L), treated as horizontal separation
    - chassis clearance (C), distance from the axle line to the bottom of the chassis (perpendicular to axle line)
- Simulates the car moving forward across the terrain with a simplified physics model.
- Uses an evolutionary algorithm (selection + crossover + mutation) to improve cars over generations.

No external dependencies (standard library only).

Usage:
    python evolve_car.py --gens 20 --pop-size 24 --seed 42

Outputs:
    - Prints per-generation stats to stdout
    - Saves the best car and run summary to results/best_car.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple


# ----------------------
# Utility helpers
# ----------------------
def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


# ----------------------
# Terrain
# ----------------------
class Terrain:
    """
    Procedural 2D terrain:
    h(x) = sum_i a_i * sin(2*pi*f_i*x + phi_i) + sum_j bump_j(x) + offset

    - Sine terms give smooth undulation
    - Bumps are Lorentzian pulses for obstacles
    - Offset ensures typical starting section is non-negative
    """

    def __init__(
        self,
        seed: int | None = None,
        n_sines: int = 5,
        n_bumps: int = 30,
        amplitude_scale: float = 1.0,
        bump_span: Tuple[float, float] = (0.0, 400.0),
    ):
        self.seed = int(seed if seed is not None else time.time_ns() & 0xFFFFFFFF)
        self.rng = random.Random(self.seed)

        self.n_sines = n_sines
        self.n_bumps = n_bumps
        self.amplitude_scale = amplitude_scale

        # Sine components
        self.amps: List[float] = []
        self.freqs: List[float] = []
        self.phases: List[float] = []
        for _ in range(self.n_sines):
            self.amps.append(self.rng.uniform(0.2, 1.6) * self.amplitude_scale)
            # Frequency in cycles per unit distance
            self.freqs.append(self.rng.uniform(0.03, 0.18))
            self.phases.append(self.rng.uniform(0.0, 2.0 * math.pi))

        # Bumps: (center_x, height, width)
        self.bumps: List[Tuple[float, float, float]] = []
        x_min, x_max = bump_span
        for _ in range(self.n_bumps):
            x0 = self.rng.uniform(x_min, x_max)
            height = self.rng.uniform(0.15, 1.4) * self.amplitude_scale
            width = self.rng.uniform(0.8, 5.0)
            self.bumps.append((x0, height, width))

        # Compute offset so initial segment is not below 0
        self.offset = 0.0
        self._compute_offset()

    def _base_height(self, x: float) -> float:
        y = 0.0
        for a, f, p in zip(self.amps, self.freqs, self.phases):
            y += a * math.sin(2.0 * math.pi * f * x + p)
        for x0, height, width in self.bumps:
            z = (x - x0) / width
            y += height / (1.0 + z * z)
        return y

    def _compute_offset(self, sample_span: Tuple[float, float] = (0.0, 200.0), samples: int = 800):
        x0, x1 = sample_span
        min_y = float("inf")
        for i in range(samples):
            x = lerp(x0, x1, i / (samples - 1))
            min_y = min(min_y, self._base_height(x))
        self.offset = -min_y + 0.0 if min_y < 0 else 0.0

    def h(self, x: float) -> float:
        return self._base_height(x) + self.offset

    def slope(self, x: float) -> float:
        # Analytic derivative for sines + bumps
        dy = 0.0
        for a, f, p in zip(self.amps, self.freqs, self.phases):
            dy += a * (2.0 * math.pi * f) * math.cos(2.0 * math.pi * f * x + p)
        for x0, height, width in self.bumps:
            z = (x - x0) / width
            dy += -2.0 * height * z / (width * (1.0 + z * z) ** 2)
        return dy

    def roughness(self, x1: float, x2: float, n: int = 25) -> float:
        if x2 <= x1:
            return 0.0
        y1 = self.h(x1)
        y2 = self.h(x2)
        dx = x2 - x1
        num = max(2, n)
        residuals: List[float] = []
        for i in range(num):
            x = x1 + dx * (i / (num - 1))
            y = self.h(x)
            # subtract linear trend across the segment
            y_lin = y1 + (y2 - y1) * ((x - x1) / (dx))
            residuals.append(y - y_lin)
        mean = sum(residuals) / len(residuals)
        var = sum((r - mean) ** 2 for r in residuals) / len(residuals)
        return math.sqrt(var)

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "n_sines": self.n_sines,
            "n_bumps": self.n_bumps,
            "amps": self.amps,
            "freqs": self.freqs,
            "phases": self.phases,
            "bumps": self.bumps,
            "offset": self.offset,
        }


# ----------------------
# Car and Simulation
# ----------------------
@dataclass
class Car:
    r1: float       # rear wheel radius
    r2: float       # front wheel radius
    length: float   # horizontal separation between wheel x-positions
    clearance: float  # perpendicular distance from axle line to bottom of chassis

    def to_dict(self) -> dict:
        return asdict(self)


def chassis_bottom_line(p1: Tuple[float, float], p2: Tuple[float, float], clearance: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Given the two hub points p1 (rear) and p2 (front), return the endpoints of the bottom-of-chassis line,
    which is parallel to the hub line but offset "downwards" by 'clearance' along the normal.
    """
    (x1, y1), (x2, y2) = p1, p2
    vx, vy = (x2 - x1), (y2 - y1)
    L = math.hypot(vx, vy)
    if L < 1e-9:
        # Degenerate; no orientation, just offset vertically
        return (x1, y1 - clearance), (x2, y2 - clearance)
    ux, uy = vx / L, vy / L
    # A left-normal is (-uy, ux); right-normal is (uy, -ux)
    nx_left, ny_left = -uy, ux
    nx_right, ny_right = uy, -ux
    # Choose the normal that points downward (negative y component)
    if ny_left < 0:
        nx, ny = nx_left, ny_left
    else:
        nx, ny = nx_right, ny_right
    # Offset both endpoints by clearance along the chosen normal
    b1 = (x1 + (-nx) * clearance * -1.0, y1 + (-ny) * clearance * -1.0)  # careful with signs
    # Actually simpler: subtract the normal scaled by clearance
    b1 = (x1 - nx * clearance, y1 - ny * clearance)
    b2 = (x2 - nx * clearance, y2 - ny * clearance)
    return b1, b2


def line_y_at_x(p1: Tuple[float, float], p2: Tuple[float, float], x: float) -> float:
    """Linear interpolation to get y on the line through p1->p2 at coordinate x."""
    (x1, y1), (x2, y2) = p1, p2
    dx = x2 - x1
    if abs(dx) < 1e-9:
        # Vertical line; return max y to avoid false collisions
        return max(y1, y2)
    t = (x - x1) / dx
    return y1 + (y2 - y1) * t


def collides_with_terrain(p1: Tuple[float, float], p2: Tuple[float, float], clearance: float, terrain: Terrain, samples: int = 50) -> bool:
    """
    Check if the bottom-of-chassis line intersects the terrain between the wheels.
    """
    b1, b2 = chassis_bottom_line(p1, p2, clearance)
    x_lo = min(b1[0], b2[0])
    x_hi = max(b1[0], b2[0])
    for i in range(samples):
        x = lerp(x_lo, x_hi, i / (samples - 1))
        y_line = line_y_at_x(b1, b2, x)
        y_ground = terrain.h(x)
        if y_ground > y_line:
            return True
    return False


def simulate(car: Car, terrain: Terrain, max_time: float = 25.0, dt: float = 0.02) -> float:
    """
    Simulate a car driving forward along the terrain.

    Simplified dynamics:
    - Wheel x-positions separated by horizontal distance 'car.length'
    - Wheel centers at (x_i, terrain.h(x_i) + r_i)
    - Chassis bottom is a line parallel to the axle line and offset by 'clearance'
    - Forward speed depends on wheel radii, local slopes, and roughness
    - Stop on chassis collision or if speed becomes too small

    Returns total distance travelled (rear wheel x-position).
    """
    x_r = 0.0
    x_f = car.length
    t = 0.0

    # Constants for speed model
    base_speed = 2.2 + 1.6 * ((car.r1 + car.r2) * 0.5)  # bigger wheels, faster
    k_slope = 1.2
    k_rough = 3.8

    total_distance = 0.0
    while t < max_time:
        y_r = terrain.h(x_r)
        y_f = terrain.h(x_f)
        p_r = (x_r, y_r + car.r1)
        p_f = (x_f, y_f + car.r2)

        # Collision with ground under chassis
        if collides_with_terrain(p_r, p_f, car.clearance, terrain, samples=40):
            break

        # Speed penalties
        slope_r = abs(terrain.slope(x_r))
        slope_f = abs(terrain.slope(x_f))
        slope_penalty = 1.0 / (1.0 + k_slope * (max(slope_r, slope_f) ** 2))
        rough = terrain.roughness(x_r, x_f, n=25)
        rough_penalty = 1.0 / (1.0 + k_rough * rough)

        # Additional stall heuristics for sharp upcoming bumps (front wheel)
        ahead = 0.25
        delta_front = terrain.h(x_f + ahead) - terrain.h(x_f)
        stall_penalty = 1.0
        if delta_front > 0.85 * car.r2:
            stall_penalty *= 0.25
        elif delta_front > 0.55 * car.r2:
            stall_penalty *= 0.5

        # Clearance margin at midpoint also affects speed
        b1, b2 = chassis_bottom_line(p_r, p_f, car.clearance)
        xm = 0.5 * (b1[0] + b2[0])
        y_line_mid = line_y_at_x(b1, b2, xm)
        gap_mid = y_line_mid - terrain.h(xm)
        gap_penalty = 1.0
        if gap_mid < car.clearance * 0.25:
            gap_penalty *= 0.5
        elif gap_mid < car.clearance * 0.15:
            gap_penalty *= 0.25

        speed = base_speed * slope_penalty * rough_penalty * stall_penalty * gap_penalty

        # Very low speed -> stop (stalled)
        if speed < 0.02:
            break

        # Integrate
        dx = speed * dt
        x_r += dx
        x_f += dx
        total_distance = x_r
        t += dt

    return total_distance


# ----------------------
# Evolutionary Algorithm
# ----------------------
CAR_BOUNDS = {
    "r1": (0.2, 1.4),
    "r2": (0.2, 1.4),
    "length": (1.2, 6.0),
    "clearance": (0.12, 1.2),
}


def random_car(rng: random.Random) -> Car:
    return Car(
        r1=rng.uniform(*CAR_BOUNDS["r1"]),
        r2=rng.uniform(*CAR_BOUNDS["r2"]),
        length=rng.uniform(*CAR_BOUNDS["length"]),
        clearance=rng.uniform(*CAR_BOUNDS["clearance"]),
    )


def mutate_car(parent: Car, rng: random.Random, sigma_scale: float = 0.15) -> Car:
    # Mutate each parameter with Gaussian noise, clipped to bounds
    def mut(val: float, lo: float, hi: float) -> float:
        scale = (hi - lo) * sigma_scale
        return clamp(val + rng.gauss(0.0, scale), lo, hi)

    return Car(
        r1=mut(parent.r1, *CAR_BOUNDS["r1"]),
        r2=mut(parent.r2, *CAR_BOUNDS["r2"]),
        length=mut(parent.length, *CAR_BOUNDS["length"]),
        clearance=mut(parent.clearance, *CAR_BOUNDS["clearance"]),
    )


def crossover(a: Car, b: Car, rng: random.Random) -> Car:
    # Blend crossover with random mixing
    def blend(va: float, vb: float, lo: float, hi: float) -> float:
        alpha = rng.uniform(-0.25, 1.25)  # allow slight extrapolation
        v = (1.0 - alpha) * va + alpha * vb
        return clamp(v, lo, hi)

    return Car(
        r1=blend(a.r1, b.r1, *CAR_BOUNDS["r1"]),
        r2=blend(a.r2, b.r2, *CAR_BOUNDS["r2"]),
        length=blend(a.length, b.length, *CAR_BOUNDS["length"]),
        clearance=blend(a.clearance, b.clearance, *CAR_BOUNDS["clearance"]),
    )


def evolve(
    gens: int = 20,
    pop_size: int = 24,
    elite_count: int = 4,
    seed: int | None = 42,
    verbose: bool = True,
):
    assert pop_size >= elite_count + 1
    rng = random.Random(seed)

    # Single fixed terrain for fair comparison across the evolution
    terrain = Terrain(seed=rng.randint(0, 10_000_000))

    # Initialize population
    population: List[Car] = [random_car(rng) for _ in range(pop_size)]
    history = []

    best_overall = None
    best_fitness_overall = -float("inf")

    for g in range(1, gens + 1):
        # Evaluate
        scored = []
        for car in population:
            fit = simulate(car, terrain, max_time=25.0, dt=0.02)
            scored.append((fit, car))
        scored.sort(key=lambda x: x[0], reverse=True)

        best_fit, best_car = scored[0]
        avg_fit = sum(f for f, _ in scored) / len(scored)
        med_fit = scored[len(scored) // 2][0]

        if verbose:
            print(f"Generation {g:02d}: best={best_fit:.2f}, avg={avg_fit:.2f}, median={med_fit:.2f}")
            bc = best_car
            print(f"  Best car: r1={bc.r1:.3f}, r2={bc.r2:.3f}, L={bc.length:.3f}, clr={bc.clearance:.3f}")

        history.append({
            "generation": g,
            "best_fitness": best_fit,
            "avg_fitness": avg_fit,
            "median_fitness": med_fit,
            "best_car": best_car.to_dict(),
        })

        if best_fit > best_fitness_overall:
            best_fitness_overall = best_fit
            best_overall = best_car

        # Create next generation (skip after last generation)
        if g == gens:
            break

        # Elitism
        next_pop: List[Car] = [car for _, car in scored[:elite_count]]

        # Produce offspring
        # Tournament selection helper
        def tournament(k: int = 3) -> Car:
            candidates = rng.sample(scored[: max(6, pop_size // 2)], k)
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]

        while len(next_pop) < pop_size:
            if rng.random() < 0.5:
                # Crossover + light mutation
                p1 = tournament(3)
                p2 = tournament(3)
                child = crossover(p1, p2, rng)
                child = mutate_car(child, rng, sigma_scale=0.08)
            else:
                # Mutate top or tournament winner
                parent = tournament(3) if rng.random() < 0.7 else scored[rng.randint(0, pop_size // 2)][1]
                child = mutate_car(parent, rng, sigma_scale=0.15)
            next_pop.append(child)

        population = next_pop

    # Save best result
    results = {
        "timestamp": int(time.time()),
        "seed": seed,
        "terrain": terrain.to_dict(),
        "best_fitness": best_fitness_overall,
        "best_car": best_overall.to_dict() if best_overall else None,
        "history": history,
    }

    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "best_car.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nEvolution complete.")
    if best_overall:
        bc = best_overall
        print(f"Best overall fitness: {best_fitness_overall:.2f}")
        print(f"Best car: r1={bc.r1:.3f}, r2={bc.r2:.3f}, L={bc.length:.3f}, clr={bc.clearance:.3f}")
        print(f"Saved results to: {out_path}")
    else:
        print("No valid car found (unexpected).")


def main():
    parser = argparse.ArgumentParser(description="Evolve a simple car to traverse random 2D terrain.")
    parser.add_argument("--gens", type=int, default=20, help="Number of generations (>=20 recommended)")
    parser.add_argument("--pop-size", type=int, default=24, help="Population size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output")
    args = parser.parse_args()

    evolve(gens=args.gens, pop_size=args.pop_size, seed=args.seed, verbose=not args.quiet)


if __name__ == "__main__":
    main()