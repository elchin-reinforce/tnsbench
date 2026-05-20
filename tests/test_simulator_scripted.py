from tnsbench.core.transcript import Transcript
from tnsbench.simulators.scripted import ScriptedSimulator
from tnsbench.tasks.task_loader import load_tasks


def test_scripted_simulator_emits_turns_in_order():
    tasks = load_tasks()
    task = next(t for t in tasks if len(t.scripted_turns) >= 3)
    sim = ScriptedSimulator()
    sim.reset(task, db_view=None)
    tr = Transcript()
    seen = []
    for _ in range(len(task.scripted_turns)):
        out = sim.step(tr, None)
        seen.append(out.user_message)
        tr.add_user(out.user_message)
    assert seen == [s.user_message for s in task.scripted_turns]


def test_scripted_no_magic_token_termination():
    tasks = load_tasks()
    task = tasks[0]
    sim = ScriptedSimulator()
    sim.reset(task, db_view=None)
    tr = Transcript()
    for s in task.scripted_turns:
        out = sim.step(tr, None)
        assert "###TRANSFER###" not in out.user_message
        tr.add_user(out.user_message)


def test_scripted_max_turns_respected():
    tasks = load_tasks()
    task = tasks[0]
    sim = ScriptedSimulator()
    sim.reset(task, db_view=None)
    tr = Transcript()
    for _ in range(len(task.scripted_turns)):
        out = sim.step(tr, None)
        tr.add_user(out.user_message)
    # After running out of script, simulator stops.
    end = sim.step(tr, None)
    assert end.should_continue is False
