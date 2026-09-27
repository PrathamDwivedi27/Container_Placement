"""Tests for ExpectedCostStrategy."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solution.expected_cost_strategy import ExpectedCostStrategy
from src.models import Container, Event, Position
from src.yard_state import YardState


LAYOUT = {"blocks": {"A": {"bays": 2, "rows": 2, "tiers": 5}}}

# Exactly two stacks, so a choice between them cannot be dodged by picking an
# empty stack elsewhere (an empty stack always costs 0 and would always win).
DUO_LAYOUT = {"blocks": {"A": {"bays": 1, "rows": 2, "tiers": 5}}}


def make_event(cid, departure, etype="DISCHARGE", timestamp="2025-01-01T00:00:00",
               port="PORT_01", weight="MEDIUM", vessel="VSL001"):
    return Event(event_id=1, timestamp=timestamp, type=etype, container_id=cid,
                 weight_class=weight, vessel_id=vessel, port_of_discharge=port,
                 departure_time=departure)


def make_container(cid, departure, port="PORT_01", weight="MEDIUM",
                   vessel="VSL001"):
    return Container(container_id=cid, weight_class=weight, vessel_id=vessel,
                     port_of_discharge=port, departure_time=departure)


@pytest.fixture
def strategy():
    s = ExpectedCostStrategy()
    s.initialize(LAYOUT, {"containers": []})
    return s


@pytest.fixture
def yard():
    return YardState(LAYOUT)


class TestValidity:
    """Placements must always satisfy the simulator's hard constraints."""

    def test_returns_valid_position_on_empty_yard(self, strategy, yard):
        event = make_event("C1", "2025-01-10T00:00:00")
        strategy.on_event(event)
        pos = strategy.place_container(yard, event)
        assert yard.is_position_valid(pos)

    def test_never_exceeds_max_tier(self, strategy, yard):
        # Fill every stack in the block to the brim but one slot.
        for bay in (1, 2):
            for row in (1, 2):
                for tier in range(1, 5):
                    yard.place_container(
                        make_container(f"F{bay}{row}{tier}", "2025-01-09T00:00:00"),
                        Position("A", bay, row, tier))
        event = make_event("C1", "2025-01-10T00:00:00")
        strategy.on_event(event)
        pos = strategy.place_container(yard, event)
        assert pos.tier == 5
        assert yard.is_position_valid(pos)

    def test_many_sequential_placements_all_valid(self, strategy, yard):
        for i in range(20):
            event = make_event(f"C{i}", f"2025-01-{10 + i % 10:02d}T00:00:00",
                               timestamp=f"2025-01-01T00:{i:02d}:00")
            strategy.on_event(event)
            pos = strategy.place_container(yard, event)
            assert yard.is_position_valid(pos), f"invalid at step {i}"
            yard.place_container(event.to_container(), pos)


class TestOrderingPreference:
    """The core rule: don't bury a container that leaves sooner."""

    def test_prefers_stacking_on_later_departing_container(self):
        strategy = ExpectedCostStrategy()
        strategy.initialize(DUO_LAYOUT, {"containers": []})
        yard = YardState(DUO_LAYOUT)
        # Row 1 holds a container leaving LATE -> safe to cover.
        # Row 2 holds one leaving SOON -> covering it causes a reshuffle.
        yard.place_container(make_container("LATE", "2025-02-01T00:00:00"),
                             Position("A", 1, 1, 1))
        yard.place_container(make_container("SOON", "2025-01-02T00:00:00"),
                             Position("A", 1, 2, 1))
        event = make_event("NEW", "2025-01-15T00:00:00")
        strategy.on_event(event)
        pos = strategy.place_container(yard, event)
        assert (pos.bay, pos.row) == (1, 1)

    def test_same_vessel_uses_port_then_weight_order(self):
        strategy = ExpectedCostStrategy()
        strategy.initialize(DUO_LAYOUT, {"containers": []})
        yard = YardState(DUO_LAYOUT)
        # Same sailing: loading goes by ascending port, then HEAVY->LIGHT.
        # PORT_09 is loaded after PORT_01, so it is the safe base.
        dep = "2025-01-20T00:00:00"
        yard.place_container(
            make_container("LATE_PORT", dep, port="PORT_09"),
            Position("A", 1, 1, 1))
        yard.place_container(
            make_container("EARLY_PORT", dep, port="PORT_01"),
            Position("A", 1, 2, 1))
        event = make_event("NEW", dep, port="PORT_05")
        strategy.on_event(event)
        pos = strategy.place_container(yard, event)
        assert (pos.bay, pos.row) == (1, 1)

    def test_empty_stack_is_preferred_over_burying_anyone(self, strategy, yard):
        """An empty stack blocks nobody, so it should always win."""
        yard.place_container(make_container("ANY", "2025-02-01T00:00:00"),
                             Position("A", 1, 1, 1))
        event = make_event("NEW", "2025-01-15T00:00:00")
        strategy.on_event(event)
        pos = strategy.place_container(yard, event)
        assert yard.get_stack_height("A", pos.bay, pos.row) == 0


class TestOverdueHandling:
    """A departure date in the past must not read as 'leaving immediately'."""

    def test_overdue_container_is_a_preferred_base(self):
        strategy = ExpectedCostStrategy()
        strategy.initialize(DUO_LAYOUT, {"containers": []})
        yard = YardState(DUO_LAYOUT)
        # Both stacks are the same height; one base is overdue (its ship sailed
        # long ago), the other leaves imminently. The overdue one is safer:
        # 57% of overdue containers do not move at all.
        yard.place_container(make_container("OVERDUE", "2025-01-01T00:00:00"),
                             Position("A", 1, 1, 1))
        yard.place_container(make_container("LEAVING_SOON", "2025-03-02T00:00:00"),
                             Position("A", 1, 2, 1))
        # Advance the clock well past the overdue container's sailing.
        event = make_event("NEW", "2025-03-05T00:00:00",
                           timestamp="2025-03-01T00:00:00")
        strategy.on_event(event)
        pos = strategy.place_container(yard, event)
        assert (pos.bay, pos.row) == (1, 1)

    def test_is_overdue_requires_clock(self, strategy):
        container = make_container("C", "2025-01-01T00:00:00")
        # Before any event is seen there is no clock, so nothing is overdue.
        assert strategy._is_overdue(container) is False
        strategy.on_event(make_event("X", "2025-01-01T00:00:00",
                                     timestamp="2025-02-01T00:00:00"))
        assert strategy._is_overdue(container) is True


class TestRobustness:
    def test_skips_phantom_padding_slots(self, strategy):
        """Sparse initial states pad stacks with "" entries (352 in test)."""
        yard = YardState(LAYOUT)
        yard.load_initial_state({"containers": [{
            "container_id": "HIGH_UP",
            "position": {"block": "A", "bay": 1, "row": 1, "tier": 3},
            "size": 20, "weight_class": "MEDIUM", "vessel_id": "VSL001",
            "port_of_discharge": "PORT_01",
            "departure_time": "2025-01-10T00:00:00"}]})
        assert yard.get_container_at("A", 1, 1, 1) == ""
        event = make_event("NEW", "2025-01-12T00:00:00")
        strategy.on_event(event)
        pos = strategy.place_container(yard, event)  # must not raise
        assert yard.is_position_valid(pos)

    def test_handles_missing_departure_time(self, strategy, yard):
        event = make_event("NEW", "")
        strategy.on_event(event)
        pos = strategy.place_container(yard, event)
        assert yard.is_position_valid(pos)

    def test_handles_unknown_vessel(self, strategy, yard):
        event = make_event("NEW", "2025-01-10T00:00:00", vessel="NOT_A_VESSEL")
        strategy.on_event(event)
        pos = strategy.place_container(yard, event)
        assert yard.is_position_valid(pos)


class TestCalibration:
    """Guard the counted constants against accidental edits."""

    def test_probabilities_are_well_formed(self):
        from solution.expected_cost_strategy import (
            OVERDUE_MIX, P_SHIP_BOUND, P_SLIP)
        assert abs(sum(OVERDUE_MIX) - 1.0) < 1e-9
        for table in (P_SHIP_BOUND, P_SLIP):
            for value in table.values():
                assert 0.0 <= value <= 1.0

    def test_no_lookahead_in_source(self):
        """The strategy must never read the event stream."""
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "solution", "expected_cost_strategy.py")
        with open(path) as fh:
            source = fh.read()
        assert "events.jsonl" not in source
        assert "read_events" not in source
