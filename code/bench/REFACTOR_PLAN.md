# SciNumBench Fabrication Engine Refactor Plan v2 (corruption_engine -> code/bench/)

Goal: provenance-aware numerical-evidence auditing. Each fabrication = injecting one violation that can be caught by a specific forensic rule,
while keeping cell-level ground truth. Fabrication <-> rule <-> detector are wired one-to-one.

## Settled decisions
- Data source: `paper_tables WHERE source='pmc' AND forensic_usable=1` (1422 tables), carrying `forensic_tags`.
- Scale: **1 instance per table** (clean or one applicable fake family), ~1422 in total.
- Families: **all 6** (SURF / GRIM / PVAL / DIST / PCT / CI).

## Data flow
paper_tables -> [Parse: keep header + typing + group n] -> [Route by tag] -> [Corrupt tier-aware, emit spans]
-> [Validate: must really change and change the right type] -> [paper-level split + cell-level provenance persisted]

## 6 fabrication families
| Family | Applicable tag | Injected violation | Detector |
|------|---------|---------|--------|
| C-SURF | any numeric table | change the last significant digit (integers stay integers) | exact recomputation / cross-table |
| C-GRIM | grim | use the real n to change the mean to a value impossible at n's decimal resolution | GRIM/GRIMMER |
| C-PVAL | pval | p contradicts the same-group effect size; supports a standalone p column | statcheck |
| C-DIST | grim/mean±SD | inflate SD so mean-2SD crosses the natural lower bound | SD/range plausibility |
| C-PCT | countpct | n/% inconsistent or percentages do not sum; supports a bare % column | count consistency |
| C-CI | ci/iqr | CI does not cover the point estimate / width contradicts SD*n | CI-mean-SE |

## One-per-table conditional assignment (seeded greedy, rare-first)
Each of the 1422 tables is assigned one condition in {clean,SURF,GRIM,PVAL,DIST,PCT,CI}, subject to eligibility constraints.
- Target: clean ~40% (~570), fake ~60% (~852).
- Assignment order: rare families first (GRIM/PVAL/CI -> DIST -> PCT -> SURF as fallback -> remaining clean),
  to avoid common families grabbing the tables that rare families could use when a table has multiple tags.
- The actual per-family counts are decided by eligibility; report the real distribution after building.

## provenance schema (new scinum_bench)
bench_id PK | arxiv_id | src_table_id(-> paper_tables.id) | source | dataset_split
| label(0/1) | corruption_family | forensic_rule
| original_grid(with header) | corrupted_grid(with header)
| provenance(JSON [{r,c,orig,new,note}]) | n_cells_changed | table_meta(JSON: group n, tags)
UNIQUE(src_table_id, dataset_split, corruption_family)

## Reproducibility
Drop global random. The master seed is fixed; per table RNG=Random(hash(src_table_id,family,master_seed)).

## Dataset split (finalized)
**Main split 60/20/20, paper-level, double-stratified, seeded**
- Order: first split papers into train/val/test -> then independently greedy-assign fabrication conditions within each split
  -> each of the three sets ~40% clean, six families in equal proportion (prevents test from leaning toward one family)
- Paper-level = a single PMCID goes into only one split (prevents leakage). Table counts ~850/285/285.
- Double stratification: label(clean/fake≈40/60) + family(6 families equal proportion) consistent across the three sets
- Serves two detector types: rule-based (GRIM/statcheck/Benford, no training) report metrics directly on test;
  learning-based (LLM probe) train on train / tune threshold on val / report on test

**OOD generalization track (overlay flag, no rebuild)**
New column `ood_role` in {train_pool, ood_test, none}:
- train_pool = train papers ∩ families in {SURF,GRIM,PVAL,DIST,clean}
- ood_test   = (val+test papers) ∩ families in {PCT,CI}   <- fabrication types unseen during training
Protocol: learning-based methods train only on train_pool and test generalization on ood_test. Papers are disjoint -> no leakage.

## Modules
code/bench/parser.py     StructuredTable (header + typing + group-n extraction)
code/bench/corruptors.py 6 families, uniformly return (grid, spans)
code/bench/build_bench.py routing + assignment + split + provenance persistence + self-check
code/bench/schema.sql

## Post-build self-check (mandatory)
- No fake is byte-identical to the original table (eliminates "labeled fake but unchanged" noise)
- provenance spans can exactly rebuild corrupted from original
- Per-family spot check: GRIM fakes really fail GRIM, PCT fakes really do not sum to 100... (fabrication <-> rule closed loop)

## Old bugs fixed
extracted_tables->paper_tables(id) | parsing dropped the header | Windows paths | random not seeded
| still labeled fake after downgrade | L2/L6-B layout assumptions did not match PMC | L3 hardcoded N=30 | no cell provenance | input() blocking
