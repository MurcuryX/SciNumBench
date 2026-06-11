# SciNumBench — Demo Data

Demo subset of the scientific-paper table numerical-fabrication detection benchmark (ICDE26). The full data lives on server1 at
`~/Backup/paper/ICDE26/data/` (splits 16M, features 5.4G, arxiv_data.db 252M).

## Files
- `splits/train_demo.jsonl` — 100 training samples (50 clean + 50 fake)
- `splits/test_demo.jsonl`  — 100 test samples (50 clean + 50 fake)
- `splits/gen_ood_mapping.jsonl` — OOD generalization track mapping (full 326 entries, small in size)
- `split_stats.json` — full dataset split statistics (src/pos/neg counts for train/val/test)

## Fields of each record
| Field | Description |
|------|------|
| example_id | unique id |
| label | 0=clean, 1=fake (numerical fabrication injected) |
| text | Caption + markdown table (model input) |
| src_table_id | source table id (-> paper_tables) |
| arxiv_id | source paper |
| provenance | cell-level ground truth: [{r,c,orig,new,is_ours,polarity}] |

Fabrication families (6 types, see code/bench/REFACTOR_PLAN.md): SURF / GRIM / PVAL / DIST / PCT / CI.
For clean records the provenance is empty; fake records list the tampered cells with their original/new values.
