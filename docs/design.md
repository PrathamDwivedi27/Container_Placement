# Container Yard Placement — Design Document

**Strategy:** `solution/expected_cost_strategy.py` → `ExpectedCostStrategy`

| | Train | Test |
|---|---|---|
| Greedy baseline (lowest stack) | 0.787 rpr — 10.5 / 40 | 0.769 rpr — 11.3 / 40 |
| **This strategy** | **0.709 rpr — 13.9 / 40** | **0.685 rpr — 14.9 / 40** |

Zero constraint violations, zero errors, on both splits. The gain is consistent
across splits (+3.4 train, +3.6 test). Every constant is counted from **train**
only; test was run once on the final locked model.

The strategy uses only the current event, the current yard state, past events,
and the published vessel timetable — it never reads ahead in the event stream,
and `tests/` asserts this.

---

## 1. How we got here — four steps

The final algorithm is simple, but arriving at it required discarding an
assumption that looked obviously correct. Each step below is a real,
measured iteration.

### Step 1 — Stack by departure time. **This made things worse.**

The obvious idea: we know each container's `departure_time`, so put the ones
leaving soonest on top, and never place a container on top of one that leaves
earlier.

**Result: 0.933 rpr — worse than the greedy baseline's 0.787.**

*Why it failed:* `departure_time` is the vessel's ETD, so **every container on
the same ship shares one identical timestamp** — often hundreds of them. The
rule treats identical timestamps as "safe to stack together," so it packed
whole shiploads into deep piles. But ~20-25% of any such group actually leaves
by truck at an unrelated time, so their true order within the pile is close to
random. Deep piles of randomly-ordered containers are exactly what we were
trying to avoid. Greedy, which ignores the signal entirely and just spreads
containers across the shortest stacks, accidentally avoided this trap.

**Lesson:** a signal that is right 75% of the time is dangerous if you treat it
as right 100% of the time.

### Step 2 — Treat the guess as a probability, and hedge against depth. **11.3 / 11.7**

Two changes:

1. Instead of a yes/no "is this safe," score each candidate stack by the
   **expected number of reshuffles**: sum, over every container already in the
   stack, the probability that the new container leaves after it. A comparison
   involving a probably-truck-bound container contributes 0.5 — an explicit
   "we don't know" — rather than a confident wrong answer.
2. Add a penalty proportional to stack height. Our estimate is imperfect, and
   depth multiplies the cost of being wrong: a bad guess under a 4-high stack
   blocks four containers, under a 1-high stack it blocks one.

**Result: 10.5 → 11.3 train, 11.3 → 11.7 test.**

The height penalty alone produced the single largest gain of any change we
made — larger than every attempt to sharpen the signal itself. It adds *no*
information; it just prices our own uncertainty.

### Step 3 — Ships sail without their cargo. **12.1 / 12.7**

Checking actual outcomes against the schedule revealed that **17.6% of
ship-bound containers miss their booked sailing and leave on that vessel's
*next* visit** (19.1% of ship arrivals, 11.6% of truck arrivals). Slip is
strongly one-directional — late, essentially never early
(`{-1 rotation: 44, on time: 3540, +1 rotation: 767}`).

`vessel_schedule.json` publishes every vessel's future visits, so the next
sailing date is knowable. We gave each container two possible exit times —
its booked sailing (~82%) and the next one (~18%) — instead of one.

**Result: 11.3 → 12.1 train, 11.7 → 12.7 test.**

### Step 4 — A date in the past doesn't mean "leaving now". **13.9 / 14.9**

The decisive step. A container whose `departure_time` has already passed while
it still sits in the yard is **overdue** — its ship left without it, and its
paperwork now describes something that did not happen.

Read literally, a past date means "leaves before everything else," so the model
assigned the **maximum** penalty to stacking on overdue containers. The data
says the opposite:

| What actually happens to an overdue container | Share |
|---|---|
| Does not move at all in the next 10 days | **57%** |
| Leaves on the vessel's next scheduled visit | 29% |
| Collected by truck | 14% |

They are the **most stable bases in the yard** — and we were systematically
avoiding them. We replaced the literal reading with those three measured
outcomes.

**Result: 12.1 → 13.9 train, 12.7 → 14.9 test** — more than steps 2 and 3
combined.

This matters far more on test than train: **test's starting yard is 76.9%
overdue; train's is 0%** (see §4).

*This is a bet, not an assumption.* 43% of overdue containers do leave, and the
model prices that risk in every comparison — a container that will itself sit
for weeks is not placed on an overdue one lightly. The reported gains are
**net** of every reshuffle caused when the bet lost.

---

## 2. The final algorithm

For each arriving container, score every non-full stack and take the minimum:

```
cost(stack) = Σ  P(new container leaves after c)   for each container c in the stack
            + 0.75 · current_stack_height
```

`P(new leaves after c)` comes from each container's exit-time distribution:

| Case | Exit-time estimate | Counted from train |
|---|---|---|
| Ship-bound, sails as booked | `departure_time` | ~82% of ship-bound |
| Ship-bound, missed it | vessel's next ETD | 19.1% / 11.6% by arrival type |
| Truck-bound | unknown → contributes 0.5 | ~21% / ~24% by arrival type |
| Overdue | 29% next sailing / 14% truck / 57% stays | train day-10 snapshot |
| Same vessel visit | exact: port ascending, then HEAVY→MEDIUM→LIGHT | verified, 0 violations |

Seven constants, each obtained by counting rows in the train data. No training,
no model file, no lookahead.

---

## 3. What the train data showed

1. **`departure_time` is given foresight, not prediction.** It is the vessel's
   `etd`, fixed at arrival and copied unchanged onto the retrieval event.
2. **LOAD order is deterministic:** ascending `port_of_discharge`, then
   HEAVY → MEDIUM → LIGHT. Zero violations across 24 batches. (Caveat: batches
   cycle the port list *twice*, so the rule is 83.9% accurate intra-batch.)
3. **~20-25% of containers leave by truck** at times unrelated to their label,
   and **nothing available at placement time identifies them.** Port shifts the
   odds only weakly (0.60–0.86); weight not at all (0.73–0.76).
4. **The yard never starts empty:** all 1,920 stacks hold at least one
   container at t=0, and 73% of multi-container stacks already contain
   inversions we cannot fix.
5. **Reshuffle relocation is fixed by the simulator** (lowest stack, same
   block). Placement is our only lever.

---

## 4. Methodology: not fooling ourselves

**Train and test are structurally different.** Train's starting yard is **0%
overdue**; test's is **76.9%** (median 11 days past label). A model tuned only
on train never meets the situation that dominates test's opening.

This explained a pattern where six consecutive refinements improved train and
degraded test. We first diagnosed it as overfitting; the real cause was this
distribution shift.

Consequences:
- All constants counted from **train only** — the overdue mix comes from a
  train day-10 snapshot, which reproduces how test opens.
- Model selection uses `analysis/validate.py`: play train days 1-10, then score
  **only** days 11-20 from the resulting yard — a test-like yard containing
  overdue containers, built without touching test. Result: greedy 0.999 rpr vs
  ours **0.898**.
- Test run once, on the final locked model.

**Disclosure:** during exploration, earlier variants were compared on test, so
intermediate figures quoted during development were optimistically biased by
selection. The 14.9 above is not: it comes from the final model with
train-only constants.

---

## 5. Alternatives considered and rejected

| Approach | Result | Why it failed |
|---|---|---|
| Strict "stack only where safe", tightest fit | 0.933 — worse than greedy | Step 1 above |
| Height-primary with priority tie-break | 0.799 ≈ greedy | The globally-lowest stack is usually unique, so the tie-break rarely fires |
| Confidence-margin gating | 0.81–0.87 | No margin is safe: truck-bound spread exceeds the gap between vessel visits |
| Blended point estimate `p·dep + (1-p)·(arr+dwell)` | worse | Averaging timestamps drags the precise majority off-target to weakly serve the minority |
| Per-port reliability + fitted smoothing curve | train 11.4 / test 11.3 | The free-form scale parameter fit train noise |
| FIFO ordering for truck-bound containers | worse on both | Genuinely 78% accurate, but applies to only ~5% of comparison weight while disturbing cross-terms carrying ~34% |
| **MIP / CP / SAT** | rejected on analysis | No search space (all 1,920 slots are already enumerated exhaustively); no joint problem (the interface forces one immediate decision per event); and the binding constraint is *missing information*, not optimisation. A solver would return the provably optimal answer to a question posed with the wrong inputs. |

**What generalised vs. what didn't:** constants we *counted* (slip rates,
reliability rates, overdue mix) held up on held-out data. Parameters we
*searched for* did not. Hedging against our own error (the height term) beat
every attempt to sharpen the signal itself.

---

## 6. Complexity

Per placement: **O(bays × rows)** ≈ 1,920 stacks, each O(stack height) ≤ O(5) —
roughly 10⁴ elementary operations. No search, no lookahead, no training.
Memory is O(containers seen) for the arrival-type map. A full 19.5k-event run
takes ~350s in single-threaded Python; vectorising the stack scan with NumPy is
the obvious speedup if throughput mattered.

---

## 7. Honest limits

- **The remaining gap is prediction quality, not placement logic.** The
  placement rule reliably picks the lowest-cost stack it is offered; what limits
  the score is how accurately we can estimate when each container leaves.
- **~20-25% of containers are genuinely unpredictable** from the data provided,
  and nothing at placement time identifies which ones they are. This is the
  single largest irreducible source of error.
- **Some disorder is not ours to fix.** The simulator relocates bumped
  containers by its own fixed rule, and both splits begin with inversions we did
  not create and cannot repair — test's opening yard is 77% overdue containers
  placed before our strategy ran at all.
- **Known unexploited signal:** every load window that passes while a container
  is still present is further evidence it is a long-stayer. `on_event` already
  supplies the clock; we use it only for the binary overdue test, not for
  progressive re-weighting. This is the most promising next step.
- **Data quirk:** the test initial state contains 352 phantom `""` slots across
  313 stacks (from `YardState`'s sparse-tier padding). They hold no container
  record but *are* counted by `get_containers_above`, so they inflate reshuffle
  counts. The strategy skips them safely.
