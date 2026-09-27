"""Expected-reshuffle-cost placement strategy.

Scores every candidate stack by the expected number of future reshuffles that
placing the incoming container there would cause, then picks the minimum.

    cost(stack) = sum over containers c already in the stack of
                      P(new container leaves after c)
                + ALPHA * current_stack_height

The first term is the expected reshuffle count: `c` only gets buried by the new
container if the new container is still sitting there when `c`'s turn comes --
i.e. if the new container leaves later. The second term is a hedge against our
own prediction error: depth multiplies the damage when the ordering estimate is
wrong, so we charge for it explicitly.

Every probability below is *counted* from the train split (never curve-fitted);
that distinction mattered empirically -- measured frequencies generalised to the
held-out split, searched parameters did not.

Exit-time model per container (three-way mixture):
  1. Ship-bound, on the labelled rotation       -- departure_time
  2. Ship-bound, slipped to the next rotation   -- next ETD from vessel_schedule
  3. Truck-bound                                -- unpredictable; contributes 0.5

Overdue containers (label already in the past) are handled separately: their
label is meaningless, so they are re-anchored onto their vessel's next *future*
rotation, with a large probability mass on "does not leave during the horizon".
On the test split 77% of the starting yard is overdue, and ~45% of those never
move again -- which makes them the safest bases in the yard, not the riskiest.

See docs/design.md for the derivation and the measurements behind each constant.
"""

import bisect
import datetime
import json
import os
from typing import Dict, List, Optional, Tuple

from src.models import Container, Event, Position
from src.placement_interface import PlacementStrategy
from src.yard_state import YardState

WEIGHT_RANK = {"HEAVY": 0, "MEDIUM": 1, "LIGHT": 2}

# P(container ends up leaving by ship | arrival type). Counted on train.
P_SHIP_BOUND = {"DISCHARGE": 0.788, "TRUCK_RECV": 0.765, "UNKNOWN": 0.780}

# P(ship-bound container slips to the vessel's next rotation | arrival type).
P_SLIP = {"DISCHARGE": 0.191, "TRUCK_RECV": 0.116, "UNKNOWN": 0.176}

# Outcome mix for a container whose label is already in the past:
# next rotation / truck trickle / does not leave within the horizon.
# Counted on a train-only snapshot (yard at train day 10, outcomes over the
# following 10 days), which mirrors how the test split starts.
OVERDUE_MIX = (0.29, 0.14, 0.57)
OVERDUE_TRUCK_DELAY_H = 107.0

ALPHA = 0.75

_FAR_FUTURE = "9999-12-31T23:59:59"


def _parse(ts: str) -> Optional[datetime.datetime]:
    try:
        return datetime.datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None


class ExpectedCostStrategy(PlacementStrategy):
    """Places each container in the stack with lowest expected reshuffle cost."""

    def initialize(self, yard_layout: dict, initial_state: dict) -> None:
        self._arrival_type: Dict[str, str] = {}
        self._now: Optional[datetime.datetime] = None
        self._vessel_etds: Dict[str, List[str]] = {}
        self._load_vessel_schedule()

    def _load_vessel_schedule(self) -> None:
        """Load the (static, given) vessel rotation table.

        Resolved relative to this file so it works from any working directory.
        """
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, os.pardir, "data", "vessel_schedule.json")
        try:
            with open(path) as fh:
                schedule = json.load(fh)
        except (OSError, ValueError):
            return
        for vessel in schedule.get("vessels", []):
            etds = sorted(r["etd"] for r in vessel.get("rotations", []))
            self._vessel_etds[vessel["vessel_id"]] = etds

    def on_event(self, event: Event) -> None:
        # Current simulation time, used to tell overdue containers from live
        # ones. Only ever reads the event being processed now -- no lookahead.
        stamp = _parse(event.timestamp)
        if stamp is not None:
            self._now = stamp
        if event.type in ("DISCHARGE", "TRUCK_RECV"):
            self._arrival_type[event.container_id] = event.type

    def _next_etd(self, vessel_id: str, departure_time: str) -> Optional[str]:
        etds = self._vessel_etds.get(vessel_id)
        if not etds:
            return None
        idx = bisect.bisect_right(etds, departure_time)
        return etds[idx] if idx < len(etds) else None

    def _first_future_etd(self, vessel_id: str) -> Optional[str]:
        etds = self._vessel_etds.get(vessel_id)
        if not etds or self._now is None:
            return None
        now = self._now.isoformat()
        idx = bisect.bisect_right(etds, now)
        return etds[idx] if idx < len(etds) else None

    def _is_overdue(self, container: Container) -> bool:
        if self._now is None or not container.departure_time:
            return False
        return container.departure_time < self._now.isoformat()

    def _exit_options(self, container: Container,
                      arrival_type: str) -> List[Tuple[str, float]]:
        """Candidate exit times with probabilities (ship-bound branch only).

        Truck-bound mass is handled by the caller, which falls back to 0.5 for
        any comparison involving it.
        """
        if self._is_overdue(container):
            p_next, p_truck, p_stay = OVERDUE_MIX
            options: List[Tuple[str, float]] = [(_FAR_FUTURE, p_stay)]
            nxt = self._first_future_etd(container.vessel_id)
            options.append((nxt if nxt else _FAR_FUTURE, p_next))
            if self._now is not None:
                truck_at = self._now + datetime.timedelta(
                    hours=OVERDUE_TRUCK_DELAY_H)
                options.append((truck_at.isoformat(), p_truck))
            else:
                options.append((_FAR_FUTURE, p_truck))
            return options

        slip = P_SLIP.get(arrival_type, P_SLIP["UNKNOWN"])
        nxt = self._next_etd(container.vessel_id, container.departure_time)
        if nxt is None:
            return [(container.departure_time, 1.0)]
        return [(container.departure_time, 1.0 - slip), (nxt, slip)]

    def _p_new_leaves_after(self, new: Container, new_type: str,
                            other: Container, other_type: str) -> float:
        """P(new container is retrieved after the one already in the stack)."""
        # Same vessel visit: the loadout order is fully determined -- ports
        # ascending, then HEAVY -> MEDIUM -> LIGHT.
        if (new.departure_time == other.departure_time
                and not self._is_overdue(new)
                and not self._is_overdue(other)):
            a = (new.port_of_discharge, WEIGHT_RANK.get(new.weight_class, 1))
            b = (other.port_of_discharge,
                 WEIGHT_RANK.get(other.weight_class, 1))
            if a > b:
                return 1.0
            if a < b:
                return 0.0
            return 0.5

        total = 0.0
        for new_at, p_new in self._exit_options(new, new_type):
            for other_at, p_other in self._exit_options(other, other_type):
                weight = p_new * p_other
                if new_at == other_at:
                    total += weight * 0.5
                elif new_at > other_at:
                    total += weight
        return total

    def place_container(self, yard_state: YardState, event: Event) -> Position:
        new = event.to_container()
        new_type = self._arrival_type.get(event.container_id, event.type)
        p_new_ship = P_SHIP_BOUND.get(new_type, P_SHIP_BOUND["UNKNOWN"])

        best_cost = None
        best_pos = None

        for block_name, info in yard_state.blocks.items():
            for bay in range(1, info.bays + 1):
                for row in range(1, info.rows + 1):
                    height = yard_state.get_stack_height(block_name, bay, row)
                    if height >= info.tiers:
                        continue

                    cost = ALPHA * height
                    for tier in range(1, height + 1):
                        cid = yard_state.get_container_at(
                            block_name, bay, row, tier)
                        # Sparse initial states pad stacks with "" placeholders
                        # (352 of them in the test split); they have no
                        # container record, so skip them.
                        if not cid:
                            continue
                        other = yard_state.get_container_info(cid)
                        if other is None:
                            continue
                        other_type = self._arrival_type.get(cid, "UNKNOWN")
                        p_other_ship = P_SHIP_BOUND.get(
                            other_type, P_SHIP_BOUND["UNKNOWN"])
                        both_ship = p_new_ship * p_other_ship
                        ordered = self._p_new_leaves_after(
                            new, new_type, other, other_type)
                        # Any comparison involving a truck-bound container has
                        # no usable ordering signal, so it costs half a move.
                        cost += both_ship * ordered + (1.0 - both_ship) * 0.5

                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        best_pos = Position(block_name, bay, row, height + 1)

        if best_pos is None:
            # Yard is completely full; no valid slot exists.
            first_block = next(iter(yard_state.blocks))
            return Position(first_block, 1, 1, 999)
        return best_pos
