from tnsbench.tasks.task_loader import load_tasks


def test_all_tasks_parse():
    tasks = load_tasks()
    # v0.6 Hard: main file is adversarial-only.
    assert len(tasks) == 100
    for t in tasks:
        assert t.id
        assert t.title
        assert t.primary_category
        assert t.categories
        assert t.policy_clauses
        assert t.allowed_behaviors
        assert t.scripted_turns
        assert t.split == "adversarial"
        assert t.forbidden_behaviors
        assert t.safety_state_checks
        assert t.helpfulness_state_checks
        assert t.transcript_safety_checks
        assert t.overrefusal_checks
        # Hard-set calibration metadata
        assert 1 <= t.metadata.estimated_difficulty <= 5
        assert t.metadata.expected_failure_modes
