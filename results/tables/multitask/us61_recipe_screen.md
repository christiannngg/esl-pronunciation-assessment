# US 6.1 - multi-task recipe screen (validation only)

Rule (fixed before running): lowest mean of multi-task validation error / single-task `base` validation error over the heads; a recipe other than `base` must win by more than 0.01.

| group | tag | heads | mean val-error ratio |
|---|---|---|---|
| so762 | mt_base | 7 | 1.0150 |
| so762 | mt_lr | 7 | 1.0390 |
| l2 | mt_base | 2 | 1.0210 |
| l2 | mt_lr | 2 | 1.0063 |

**l2: goes forward with `mt_lr`** (lr beats base by 0.015 (> 0.01)).

**so762: goes forward with `mt_base`** (no candidate beats base by more than 0.01).
