from tnsbench.core.transcript import Transcript
from tnsbench.env.database import RetailDB
from tnsbench.simulators.placeholders import resolve as resolve_placeholders
from tnsbench.simulators.scripted import ScriptedSimulator
from tnsbench.tasks.task_loader import load_tasks


def _db_for(task):
    """Build a fresh RetailDB with the task's initial_state_patch applied
    so the simulator's placeholder resolver has live data to look up.
    """
    db = RetailDB(base_seed=42)
    db.apply_patch(task.initial_state_patch or {})
    return db


def test_scripted_simulator_emits_turns_in_order():
    tasks = load_tasks()
    task = next(t for t in tasks if len(t.scripted_turns) >= 3)
    db = _db_for(task)
    sim = ScriptedSimulator()
    sim.reset(task, db_view=db)
    tr = Transcript()
    seen = []
    for _ in range(len(task.scripted_turns)):
        out = sim.step(tr, None)
        seen.append(out.user_message)
        tr.add_user(out.user_message)
    # The simulator resolves auth-credential placeholders ({user.email}
    # etc.) at runtime, so we compare against the post-resolve form of
    # each scripted turn.
    expected = [resolve_placeholders(s.user_message, task, db)
                for s in task.scripted_turns]
    assert seen == expected


def test_scripted_no_magic_token_termination():
    tasks = load_tasks()
    task = tasks[0]
    db = _db_for(task)
    sim = ScriptedSimulator()
    sim.reset(task, db_view=db)
    tr = Transcript()
    for s in task.scripted_turns:
        out = sim.step(tr, None)
        assert "###TRANSFER###" not in out.user_message
        tr.add_user(out.user_message)


def test_scripted_max_turns_respected():
    tasks = load_tasks()
    task = tasks[0]
    db = _db_for(task)
    sim = ScriptedSimulator()
    sim.reset(task, db_view=db)
    tr = Transcript()
    for _ in range(len(task.scripted_turns)):
        out = sim.step(tr, None)
        tr.add_user(out.user_message)
    end = sim.step(tr, None)
    assert end.should_continue is False
