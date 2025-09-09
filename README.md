Evolving Cars on Terrain (Browser)
==================================

A small interactive demo that evolves simple "cars" so they roll as far as possible over hilly terrain. It runs entirely in the browser:

- Physics, terrain, and evolution are written in Python and executed via Pyodide (Python compiled to WebAssembly).
- Rendering and UI are written in vanilla JavaScript on an HTML5 canvas.

What you see: in each attempt a single car with two wheels and a rigid chassis tries to drive to the right. When the attempt ends, the next car is created by mutating the best one found so far, with occasional fully random cars to explore the space.

Contents
- Getting started
- Controls and panels
- How the car model works
- How the “genetic algorithm” works
- Project layout
- Customization hints

Getting started
- Requirements:
  - A local static file server (to allow the browser to fetch the Python file).
  - Internet connection (Pyodide is loaded from a CDN).
- Quick run with Python:
  - Python 3.x: from the repo root run: python3 -m http.server 8000
  - Open http://localhost:8000 in your browser.
- Quick run with Node (alternative):
  - npm i -g http-server
  - http-server -p 8000
  - Open http://localhost:8000
- Open index.html if you already serve files from elsewhere.

Controls and panels
- Play starts the simulation loop; Stop pauses it.
- Speed changes how many physics steps are simulated per animation frame (1–30x).
- Reset Terrain generates a fresh random terrain and restarts evolution.
- Attempt shows the current trial index (1-based).
- Best Distance is the farthest any car has travelled so far (front wheel position).
- Current Distance is the current car’s progress.
- Evolution panel:
  - Lineage: whether this car was Random init, Mutated from best, or a Random exploration.
  - Parent Best Dist: the best distance achieved before this attempt.
  - Genes: some of the parameters that define the current car.

How the car model works
The world is a smooth 2D heightfield made from a few sine waves. For a position x:
- height(x) returns the ground height.
- slope(x) returns dy/dx, used to make uphill segments harder.

Each car has:
- Two wheels:
  - r_back, r_front: wheel radii.
  - The wheels are always on the terrain surface (no suspension model).
  - Both wheels are driven with the same nominal angular speed omega.
- Geometry:
  - wheelbase: distance between wheels.
  - body_base_ratio: base length of the chassis as a fraction of wheelbase (clamped to a safe range).
  - body_height: a characteristic height used by the legacy fallback renderer.
  - A triangle strip that forms the chassis silhouette relative to the axle line:
    - strip_kind: one of several patterns used to seed vertices.
    - strip_x_norm: interior vertex x-positions normalized to (-0.5, 0.5) across the axle.
    - strip_y: corresponding vertex heights.
  - For compatibility, a simple tri_heights array is also kept and used by a fallback renderer.

Motion model (fast, approximate):
- Compute ground contact for back and front wheel; the car’s axle line is the segment between those two contact points (offset up by the radii).
- Orientation phi is the angle of that axle line (atan2 of the wheel contact heights).
- Each wheel has an effective forward speed:
  - v_back = max(0, omega * r_back - c_penalty * max(0, slope_back))
  - v_front = max(0, omega * r_front - c_penalty * max(0, slope_front))
- The car’s forward speed is v = clamp(min(v_back, v_front), 0, v_max).
  - Very steep local slopes cause traction loss (speed goes to 0).
- Chassis–ground interaction:
  - The chassis triangle strip is transformed into world space and sampled along edges/diagonals.
  - If the body would penetrate the ground, forward motion is reduced; above a threshold, motion stops entirely.
- Termination:
  - Attempt ends if stuck for a short time, after a time limit, or near the end of the terrain.

How the “genetic algorithm” works
This is a minimal evolutionary hill-climber (a (1+1) strategy) with random restarts:
- Genome (genes):
  - r_back, r_front: wheel sizes.
  - wheelbase: distance between wheels; must be at least 1.1 * (r_back + r_front) to avoid overlap.
  - body_base_ratio: base length of the chassis relative to wheelbase.
  - body_height: baseline height used by a legacy/fallback renderer.
  - omega: nominal angular wheel speed (both wheels).
  - Chassis strip:
    - strip_kind: how the initial interior vertices are arranged (e.g., uniform_zigzag, center_cluster, diamond_center, split_clusters, arch).
    - strip_x_norm: strictly increasing interior x positions in (-0.5, 0.5).
    - strip_y: alternating up/down heights at those positions.
  - Legacy fallback:
    - n_tris and tri_heights (still carried for compatibility with the fallback renderer).
- Initialization:
  - Start with a random car.
- Fitness:
  - Distance travelled by the front wheel (farther is better).
- Selection and reproduction:
  - After each attempt, if the current distance exceeds the best so far, that car becomes the best.
  - The next candidate is usually a mutation of the best; sometimes a fresh random car is used to escape local optima.
    - By default: mutate ~80% of the time, explore completely random ~20% of the time.
- Mutation:
  - Apply small Gaussian noise to continuous genes within bounds (radii, wheelbase, ratios, omega).
  - Maintain constraints (e.g., wheelbase >= 1.1 * (r_back + r_front)).
  - Chassis triangle strip:
    - Occasionally rebuild from scratch with a new pattern.
    - Otherwise jitter interior vertices slightly; sometimes add or remove one; enforce strictly increasing x and alternating signs for a zigzag-like shape.
    - Clamp heights and ensure a minimum magnitude so the shape remains visible and engages with terrain.

The result is a simple but effective search: cars that roll farther are kept, and their parameters get nudged to try to roll even farther. The occasional fully random car gives diversity and prevents the system from getting stuck permanently.

Project layout
- index.html: page structure, canvas, and controls; loads Pyodide from CDN.
- static/styles.css: UI and canvas styling.
- static/app.js: draws terrain and car, handles UI, and talks to the Python simulator via Pyodide.
- static/py/sim.py: Python simulation code (terrain, kinematics, evolution).

Customization hints
All the core knobs live in static/py/sim.py:
- Evolution rates:
  - In Simulator._init_new_car: adjust the exploration probability (random vs mutated-from-best).
  - In CarParams.mutated: change the scale parameter or the probabilities for adding/removing interior vertices.
- Physics/termination:
  - In Simulator._step_once: c_penalty, speed cap, traction slope threshold, stuck/time limits.
  - MIN_WHEELBASE_FACTOR sets basic wheel clearance.
- Terrain:
  - Terrain.__init__ controls base height, amplitudes, and frequencies of the rolling hills.

Notes
- This is an educational demo, not a physically accurate vehicle model. It’s intentionally simple so the effect of evolution is easy to see.
- Pyodide is fetched from a CDN; if you need offline use, bundle Pyodide locally or keep an internet connection during development.