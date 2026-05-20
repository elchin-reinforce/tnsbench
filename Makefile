.PHONY: tasks lint test run-safe run-unsafe report smoke clean

tasks:
	python3 -m tnsbench.cli generate-tasks

lint:
	python3 -m tnsbench.cli validate-tasks

test:
	python3 -m pytest tests/ -x

run-safe:
	python3 -m tnsbench.cli run --agent mock_safe --simulator scripted --tasks all --trials 1 --out results/mock_safe.jsonl

run-unsafe:
	python3 -m tnsbench.cli run --agent mock_unsafe --simulator scripted --limit 20 --trials 1 --out results/mock_unsafe.jsonl

report:
	python3 -m tnsbench.cli report results/mock_safe.jsonl --out results/mock_safe_report.md
	python3 -m tnsbench.cli report results/mock_unsafe.jsonl --out results/mock_unsafe_report.md

smoke: tasks lint test run-safe run-unsafe report

clean:
	rm -rf results/*.jsonl results/*.md results/*.json
