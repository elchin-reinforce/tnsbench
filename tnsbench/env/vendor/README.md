# vendor/ — base retail data

This directory holds the **base** retail data used by TnSBench. It is intentionally
written in the τ-bench retail data shape so that upstream τ-bench files can be
dropped in unchanged.

## Provenance

The files shipped here are **τ-bench-shaped** approximations created for TnSBench.
They are internally consistent and pass the linter, but they are **not the
verbatim upstream files** from `sierra-research/tau-bench`. To run TnSBench against
the true upstream retail dataset:

1. Clone `https://github.com/sierra-research/tau-bench`.
2. Copy `tau_bench/envs/retail/data/{users,products,orders}.json` over the files
   in `tau_bench_retail/`.
3. Re-run `python3 -m tnsbench.cli generate-tasks` to remap task IDs.

The overlay (protected attributes, paired-bias users, injection payloads,
suspicious-history flags) lives in [tnsbench/env/overlays.py](../overlays.py) and
applies on top of whichever base data is loaded.

## Files

- `users.json`     — list of user records (τ-bench shape)
- `products.json`  — product catalog
- `orders.json`    — order history

## License & attribution

τ-bench is published under its own license. If you replace the placeholder
files with upstream data, copy the upstream LICENSE next to it.
