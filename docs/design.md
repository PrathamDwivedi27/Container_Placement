# Container Yard Placement — Design Document (DRAFT — pending final numbers)

_This is a working draft being filled in while final tuning runs complete. Sections
marked TODO will be finalized with exact train/test numbers before submission._

## 1. Data analysis (train set)

Empirically verified on `data/train/events.jsonl` (20,592 events):

- `departure_time`, `vessel_id`, and `port_of_discharge` are **identical** between a
  container's arrival event (DISCHARGE/TRUCK_RECV) and its eventual retrieval event
  (LOAD/TRUCK_DLVR) — these are fixed, known-in-advance attributes, not predictions.
- **LOAD dispatch order is fully deterministic** and was verified with zero
  violations across every sampled multi-container loadout batch: within a
  `(vessel_id, departure_time)` batch, containers are unloaded grouped by
  **ascending `port_of_discharge`**, and within each port group, **HEAVY → MEDIUM →
  LIGHT**. This gives a perfect composite retrieval-order key for the ~75-80% of
  containers that ultimately leave via LOAD.
- `departure_time` is a **noisy** proxy for actual retrieval time. Median
  arrival-to-retrieval offsets are small (a few hours) but the tail is heavy:
  roughly 10-15% of containers deviate from their nominal ETD by more than a week
  (vessel rotations can slip), and TRUCK_DLVR-bound containers (~20-25% of arrivals)
  carry a `departure_time` that is frequently just an inherited/booked label
  unrelated to the actual truck pickup time (pickups are effectively unscheduled).
- Consequence: many containers that share an **identical** priority key (same
  vessel visit) have, in reality, **unordered / near-random** relative retrieval
  times. Treating a tied key as "safe to stack arbitrarily deep" is actively harmful
  — a naive "trust departure_time completely" policy performed *worse* than the
  lowest-stack-first greedy baseline in initial experiments (see §3).

## 2. Related work and adapted approach

Primary references (container-yard placement/stacking, last ~5 years):

- Güven, C. & Eliiyi, D.T. (2019). "Modelling and optimisation of online container
  stacking with operational constraints." *Maritime Policy & Management*, 46(2),
  234–249. Rule-based / attribute-based online stacking policies driven by
  container attributes (destination, weight, expected departure time).
- [Author(s)] (2022). "GMM-based online optimization for container stacking in port
  container terminals." *Computers & Industrial Engineering*. Generalizes the
  classical vertical stacking policy (Kim & Park, 2003) into data-driven priority
  *classes*, with proportional space reservation per class and best-fit placement
  within a class.

**Adaptation.** This benchmark has no continuous weight signal to cluster (only 3
fixed `weight_class` buckets), so the GMM/clustering step from the 2022 paper adds
nothing here. What the benchmark *does* provide — and what neither baseline uses —
is the deterministic composite retrieval-order key described in §1. We substitute
that key for the papers' data-driven weight class, and keep their core mechanism:
**keep every stack's retrieval order non-decreasing bottom-to-top**, so whichever
container leaves soonest stays on top and nothing blocks it.

## 3. Algorithm

TODO — final precedence rule and depth-cap parameter, pending sweep results.

## 4. Complexity

O(bays × rows) ≈ O(1,920) per placement decision; O(1) per slot in the common case,
O(stack height) ≤ O(5) only when no safe slot exists. No lookahead search, no
training, no mutable state beyond the yard itself.

## 5. Results

TODO — train and test reshuffles/retrieval, compared to Random and Greedy baselines.

## 6. Trade-offs and alternatives considered

- TODO
