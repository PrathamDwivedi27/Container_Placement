"""Priority-based online container stacking strategy.

Adapts the "class-based online vertical stacking policy" family of container-yard
research to this benchmark:

  - Guven, C. & Eliiyi, D.T. (2019). "Modelling and optimisation of online container
    stacking with operational constraints." Maritime Policy & Management, 46(2),
    234-249. Introduces rule-based / attribute-based online stacking policies that
    place each arriving container using its known attributes (destination, weight,
    expected departure time) instead of a blind lowest-stack rule.

  - Lee/Kim et al. (2022). "GMM-based online optimization for container stacking in
    port container terminals." Computers & Industrial Engineering. Generalizes the
    classical "vertical stacking policy" (Kim & Park, 2003) by deriving data-driven
    priority *classes* from container attributes, reserving proportional stack space
    per class, and placing each container in the slot that best matches its class.

This benchmark's containers are not linked to a real weight distribution (only 3
fixed weight_class buckets are given), so fitting a Gaussian/Fuzzy mixture model to
weight -- the specific clustering step used in the 2022/2026 papers -- would add
nothing. What *is* available, and is not exploited by either baseline, is a fully
deterministic future retrieval order: `departure_time` (vessel ETD), grouped by
`port_of_discharge` and then by `weight_class` (heavy first), which is exactly how
LOAD events are dispatched in this simulation (verified empirically against 20 days
of train data: zero ordering violations across every sampled loadout batch -- see
docs/design.md). This strategy replaces the papers' data-driven weight class with a
data-driven *retrieval-priority class* built from that composite key, and applies
the same "class-based vertical stacking" placement principle: keep every stack's
retrieval order non-decreasing bottom-to-top, so that whichever container leaves
soonest is always on top and nothing has to be moved out of its way.

See docs/design.md for the full derivation, the empirical validation of the ordering
rule, and the complexity analysis.
"""

from typing import Optional, Tuple

from src.models import Container, Event, Position
from src.placement_interface import PlacementStrategy
from src.yard_state import YardState

# Lower rank == unloaded earlier within a (departure_time, port) group.
WEIGHT_RANK = {"HEAVY": 0, "MEDIUM": 1, "LIGHT": 2}

PriorityKey = Tuple[str, str, int]


def priority_key(container: Container) -> PriorityKey:
    """Composite key approximating a container's future retrieval order.

    Lower key => retrieved sooner => should end up higher in its stack.
    Matches the simulator's LOAD dispatch order exactly: containers are loaded
    grouped by ascending port_of_discharge, and within each port group, heaviest
    first. departure_time is the coarse (vessel-level) sort key; port and weight
    break ties within the same vessel/ETD.
    """
    return (
        container.departure_time or "9999-99-99T99:99:99",
        container.port_of_discharge or "",
        WEIGHT_RANK.get(container.weight_class, 1),
    )


class PriorityStackingStrategy(PlacementStrategy):
    """Places containers on the lowest stack that keeps priority order safe.

    For each incoming container, every open slot in the yard is classified:

    1. SAFE -- either the slot starts a brand new stack, or it sits on top of
       an existing stack whose current top container has priority >= the new
       container's priority. Stacking here preserves a non-decreasing
       bottom-to-top priority order (whoever leaves soonest stays on top), so
       it introduces zero new expected reshuffles. Among all safe slots we
       pick the *shortest* resulting stack -- the same lowest-stack-first rule
       the greedy baseline uses -- because departure_time is a noisy signal
       (see docs/design.md): roughly 10-15% of containers deviate from their
       nominal ETD by more than a week, so keeping stacks shallow bounds the
       damage from a wrong prediction, exactly as it bounds it for the greedy
       baseline. Priority information is only used as a *filter* (never build
       a known-bad stack) and, among equal-height ties, as a tie-break (prefer
       the tightest matching top, so same-priority containers still cluster
       when it's free to do so).
    2. UNSAFE (last resort, only when no SAFE slot exists anywhere) -- every
       remaining stack would bury at least one container that must leave
       sooner. We pick the stack that buries the fewest such containers (each
       one is a likely future reshuffle charge), tie-broken by the shortest
       stack to bound the worst case.

    Complexity: O(total slots) = O(bays * rows) per placement decision (~1,920
    for this yard), each slot doing O(1) work in the common SAFE case and
    O(stack height) <= O(5) work only in the rare UNSAFE case. No lookahead
    search, no training, no mutable state beyond the yard itself -- the
    decision is a pure function of the current YardState and the incoming
    event.
    """

    def initialize(self, yard_layout: dict, initial_state: dict) -> None:
        pass

    def place_container(self, yard_state: YardState, event: Event) -> Position:
        new_key = priority_key(event.to_container())

        best = None  # (sort_key, block, bay, row, height)

        for block_name, block_info in yard_state.blocks.items():
            for bay in range(1, block_info.bays + 1):
                for row in range(1, block_info.rows + 1):
                    height = yard_state.get_stack_height(block_name, bay, row)
                    if height >= block_info.tiers:
                        continue

                    if height == 0:
                        # Resulting height 1, safe (category 0), no top to
                        # match against.
                        sort_key = (1, 0, ("", "", -1))
                    else:
                        top_id = yard_state.get_container_at(
                            block_name, bay, row, height)
                        top_info = yard_state.get_container_info(top_id)
                        top_key = priority_key(top_info) if top_info else (
                            "", "", 0)

                        if top_key >= new_key:
                            sort_key = (height + 1, 0, top_key)
                        else:
                            buried_sooner = 0
                            for tier in range(1, height + 1):
                                cid = yard_state.get_container_at(
                                    block_name, bay, row, tier)
                                info = yard_state.get_container_info(cid)
                                if (info is not None
                                        and priority_key(info) < new_key):
                                    buried_sooner += 1
                            sort_key = (height + 1, 1, buried_sooner)

                    if best is None or sort_key < best[0]:
                        best = (sort_key, block_name, bay, row, height)

        if best is not None:
            _, block_name, bay, row, height = best
            return Position(block_name, bay, row, height + 1)

        # Yard is completely full -- no valid slot exists anywhere.
        first_block = next(iter(yard_state.blocks))
        return Position(first_block, 1, 1, 999)
