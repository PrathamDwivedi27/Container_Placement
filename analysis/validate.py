"""Train-internal validation harness.

Runs train days 1-10 to build a realistic mid-simulation yard (one that
contains overdue containers, the way the test split's starting yard does),
then scores ONLY days 11-20 from that yard. Lets us compare strategies
without ever touching the test split.
"""
import datetime, json, sys, time
sys.path.insert(0, '.')
from src.yard_state import YardState
from src.event_reader import read_events
from src.simulator import Simulator


def split_events(path, days=10):
    ev = read_events(path)
    t0 = datetime.datetime.fromisoformat(ev[0].timestamp)
    cut = t0 + datetime.timedelta(days=days)
    warm = [e for e in ev if datetime.datetime.fromisoformat(e.timestamp) <= cut]
    score = [e for e in ev if datetime.datetime.fromisoformat(e.timestamp) > cut]
    return warm, score


def validate(make_strategy, label, days=10):
    layout = json.load(open('data/yard_layout.json'))
    initial = json.load(open('data/train/initial_state.json'))
    warm, score = split_events('data/train/events.jsonl', days)

    yard = YardState(layout)
    yard.load_initial_state(initial)
    strategy = make_strategy()
    strategy.initialize(layout, initial)

    t0 = time.time()
    Simulator(yard, strategy, verbose=False).run(warm)   # warm-up, not scored
    stats = Simulator(yard, strategy, verbose=False).run(score)
    print(f"[validation d{days+1}-20] {label}: rpr={stats.reshuffles_per_retrieval:.4f} "
          f"quant={stats.quantitative_score():.1f} viol={stats.hard_constraint_violations} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return stats


if __name__ == '__main__':
    from src.baseline_greedy import GreedyStrategy
    from solution.expected_cost_strategy import ExpectedCostStrategy
    validate(GreedyStrategy, "greedy")
    validate(ExpectedCostStrategy, "expected-cost (overdue-aware, train-only constants)")
