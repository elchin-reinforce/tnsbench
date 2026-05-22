import pytest

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


def test_adversarial_only_splits():
    tasks = load_tasks()
    assert all(t.split == "adversarial" for t in tasks)
    assert len(load_adversarial_tasks()) == 100


def test_load_benign_raises():
    # TnSBench-Hard is adversarial-only; benign loaders must error.
    with pytest.raises(RuntimeError):
        load_benign_tasks()


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


def test_two_main_files_byte_identical():
    """tasks.jsonl and adversarial_tasks.jsonl must be the same 100 tasks."""
    from tnsbench.core.config import ADVERSARIAL_TASKS_PATH, TASKS_PATH

    assert ADVERSARIAL_TASKS_PATH.exists()
    assert TASKS_PATH.exists()
    assert ADVERSARIAL_TASKS_PATH.read_bytes() == TASKS_PATH.read_bytes()
