# Container Yard Placement — Design Document

**Strategy:** `solution/expected_cost_strategy.py` → `ExpectedCostStrategy`

| | Train | Test |
|---|---|---|
| Greedy baseline (lowest stack) | 0.787 rpr — 10.5 / 40 | 0.769 rpr — 11.3 / 40 |
| **This strategy** | **0.709 rpr — 13.9 / 40** | **0.685 rpr — 14.9 / 40** |

Zero constraint violations and zero errors on both splits. On test this is 6,610
reshuffles against the baseline's 7,416 over the same 9,647 retrievals.

---

## 1. Approach

A container is reshuffled when something is stacked on top of it and has not
left by the time it is needed. Placement therefore comes down to one question:
*of everything already in this stack, how much of it will still be waiting when
the new container leaves?*

For each arrival, every non-full stack is scored and the lowest wins:

```
cost(stack) = Σ  P(new container leaves after c)   for each container c in the stack
            + 0.75 · current_stack_height
```

The first term is the expected number of reshuffles the placement causes. The
second is a hedge: the exit-time estimate is imperfect, and depth multiplies the
cost of a wrong guess — a bad placement under a four-high stack blocks four
containers, under a one-high stack it blocks one.

`P(new leaves after c)` is derived from each container's estimated exit time:

| Case | Exit-time estimate | Frequency (train) |
|---|---|---|
| Ship-bound, sails as booked | `departure_time` | ~82% of ship-bound |
| Ship-bound, misses its sailing | vessel's next ETD | 19.1% / 11.6% by arrival type |
| Truck-bound | not predictable — contributes 0.5 | ~21% / ~24% by arrival type |
| Overdue (see §2, step 4) | 29% next sailing / 14% truck / 57% stays | train day-10 snapshot |
| Same vessel visit | port ascending, then HEAVY→MEDIUM→LIGHT | exact |

Seven constants, each obtained by counting occurrences in the training data.
There is no training step and no model file. At runtime the strategy uses only
the current event, the current yard state, events already seen, and the
published vessel timetable.

---

## 2. Development path

### Step 1 — Order by departure time

The natural first attempt: place containers leaving soonest on top, and never
cover a container that leaves earlier.

**Result: 0.933 rpr — worse than the 0.787 baseline.**

`departure_time` is the vessel's ETD, so every container on a sailing carries an
identical timestamp, often several hundred of them. The rule read identical
timestamps as interchangeable and packed whole shiploads into deep stacks. But
roughly a quarter of any such group actually leaves by truck at an unrelated
time, so the true order inside those stacks was close to random. The greedy
baseline, which ignores the signal and simply spreads containers across the
shortest stacks, avoids this by accident.

### Step 2 — Treat the estimate as a probability, and penalise depth

Two changes: score stacks by *expected* reshuffles rather than a binary
safe/unsafe test, with a probably-truck-bound comparison contributing 0.5
instead of a confident wrong answer; and add a penalty proportional to stack
height.

**Result: 11.3 train, 11.7 test.**

The height penalty was the larger of the two effects, and it adds no information
at all — it only prices the uncertainty in our own estimate.

### Step 3 — Account for missed sailings

Comparing outcomes against the schedule showed that **17.6% of ship-bound
containers miss their booked sailing and leave on that vessel's next visit**
(19.1% of ship arrivals, 11.6% of truck arrivals). The slip is one-directional:
late, essentially never early (`{-1: 44, on time: 3540, +1: 767}`).

Since `vessel_schedule.json` lists every future visit, each container was given
two candidate exit times — its booked sailing and the following one — weighted
by those frequencies.

**Result: 12.1 train, 12.7 test.**

### Step 4 — Handle containers whose sailing has already passed

A container whose `departure_time` has passed while it is still in the yard is
**overdue**: its ship left without it, so the date no longer describes anything
that will happen.

Taken literally, a past date means "leaves before everything else", so the model
was applying the maximum penalty to covering these containers. The data shows
the opposite:

| Outcome for an overdue container | Share |
|---|---|
| Does not move within the next 10 days | **57%** |
| Leaves on the vessel's next scheduled visit | 29% |
| Collected by truck | 14% |

They are the most stable bases available, and the model had been avoiding them.
Replacing the literal reading with these three measured outcomes produced the
largest single gain of the four steps.

**Result: 13.9 train, 14.9 test.**

This is a weighted bet rather than an assumption — 43% of overdue containers do
leave, and that risk is priced into every comparison, so a container that will
itself sit for weeks is not placed on an overdue one lightly.

---

## 3. Patterns in the training data

1. **`departure_time` is supplied, not predicted.** It is the vessel's `etd`,
   fixed on arrival and copied unchanged onto the retrieval event.
2. **Ship loading order is deterministic:** ascending `port_of_discharge`, then
   HEAVY → MEDIUM → LIGHT, with no exceptions across 24 sampled batches. Batches
   cycle the port list twice, so the rule is 83.9% accurate within a batch.
3. **20-25% of containers leave by truck** at times unrelated to their label,
   and nothing available at placement time identifies which ones. Port shifts
   the odds only slightly (0.60–0.86) and weight class not at all (0.73–0.76).
4. **The yard never starts empty.** All 1,920 stacks hold at least one container
   at t=0, and 73% of multi-container stacks already contain inversions.
5. **Relocation during retrieval is controlled by the simulator** (lowest stack,
   same block), so placement is the only available lever.

---

## 4. Validation

Train and test begin from structurally different yards: **train's starting yard
contains no overdue containers, test's is 76.9% overdue** (median 11 days past
label). A model assessed only on train never encounters the condition that
dominates test's opening, and several intermediate variants improved on train
while regressing on test for this reason.

Model selection therefore used a train-internal split (`analysis/validate.py`):
play train days 1-10, then score only days 11-20 from the resulting yard, which
contains overdue containers in the way test's does. On that split the baseline
scores 0.999 rpr against this strategy's 0.898. All constants are counted from
training data, and the test set was run once, on the final configuration.

### Why both strategies score higher on test

The baseline and this strategy both gain moving from train to test (10.5 → 11.3
and 13.9 → 14.9). The cause is the dataset, not the method:

| | Train | Test |
|---|---|---|
| Median dwell time | 0.96 days | 0.80 days |
| Arrivals per day | 526 | 494 |
| **Containers arriving during a typical stay** | **503** | **396** |

A container is only reshuffled if something is stacked on it before it leaves,
so the relevant quantity is how many containers arrive during its stay — about
21% fewer in test. Scores are therefore comparable only within a split.

---

## 5. Alternatives considered

| Approach | Outcome |
|---|---|
| Strict safe/unsafe placement, tightest fit | 0.933 — worse than baseline (§2, step 1) |
| Height-first with priority as tie-break | 0.799 — the lowest stack is usually unique, so the tie-break rarely applies |
| Confidence-margin gating on the time gap | 0.81–0.87 — no margin is wide enough, since truck-bound spread exceeds the interval between vessel visits |
| Blending exit-time estimates into one value | Worse — averaging pulls the accurate majority off-target to accommodate the minority |
| Per-port reliability with a fitted smoothing curve | Train 11.4, test 11.3 — the fitted parameter matched training noise |
| FIFO ordering for truck-bound containers | Worse — the signal is 78% accurate but applies to only ~5% of comparison weight |
| MIP / CP / SAT | Not applicable: all 1,920 stacks are already enumerated exhaustively, the interface requires one decision per arrival so there is no joint problem to formulate, and the binding constraint is unavailable information rather than search |

Counted frequencies transferred to held-out data; searched parameters did not.

---

## 6. Complexity

Per placement decision: **O(bays × rows)** ≈ 1,920 stacks, each costing
O(stack height) ≤ O(5) — approximately 10⁴ elementary operations, with no search
or lookahead. Memory is O(containers seen) for the arrival-type map. A full
19,508-event run takes roughly 350 seconds in single-threaded Python;
vectorising the stack scan would be the natural optimisation if throughput
mattered.

---

## 7. Related work

The scoring rule used here belongs to the **reshuffle-index** family of stacking
heuristics, in which each candidate slot is scored by the expected number of
future relocations it would cause, rather than by a structural rule such as
lowest-stack. Đurašević and Dumić (2021) benchmark rules of this type —
including Min-Max and Reshuffle Index — for the container relocation problem.
Güven and Eliiyi (2019) address the closer variant of *online* stacking, where
each container must be placed on arrival using its known attributes.

Both assume the retrieval order is known. The main departure here is that
retrieval order is treated as uncertain and estimated probabilistically from
observed frequencies, with an explicit depth penalty reflecting confidence in
that estimate.

- Đurašević, M. & Dumić, M. (2021). *Automated Design of Heuristics for the
  Container Relocation Problem.* arXiv:2107.13313
- Güven, C. & Eliiyi, D.T. (2019). *Modelling and optimisation of online
  container stacking with operational constraints.* Maritime Policy &
  Management, 46(2), 234–249
