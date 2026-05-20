from tnsbench.core.config import REQUIRED_TASK_DISTRIBUTION, TOTAL_TASKS
from tnsbench.tasks.lint import lint_tasks, summarize_distribution
from tnsbench.tasks.task_loader import (
    load_adversarial_tasks,
    load_benign_tasks,
    load_tasks,
)


def test_total_count():
    tasks = load_tasks()
    assert len(tasks) == TOTAL_TASKS


def test_split_counts():
    assert len(load_adversarial_tasks()) == 100
    assert len(load_benign_tasks()) == 20


def test_distribution_exact():
    tasks = load_tasks()
    d = summarize_distribution(tasks)
    for k, v in REQUIRED_TASK_DISTRIBUTION.items():
        assert d.get(k) == v, f"category {k}: got {d.get(k)} expected {v}"


def test_lint_clean():
    tasks = load_tasks()
    ok, errors = lint_tasks(tasks)
    assert ok, errors


def test_paired_bias_bidirectional():
    tasks = load_tasks()
    by_id = {t.id: t for t in tasks}
    paired = [t for t in tasks if t.metadata.paired_task_id]
    assert paired, "expected paired bias tasks"
    for t in paired:
        pid = t.metadata.paired_task_id
        assert pid in by_id
        assert by_id[pid].metadata.paired_task_id == t.id
