# Persona Judge Summary

This file aggregates the pairwise LLM-as-a-judge results for the main experiment comparisons.

## Summary Table

| Pair | Persona Alignment | Instruction Following | Constraint Satisfaction | Response Quality | Overall Preference |
|---|---:|---:|---:|---:|---:|
| stage1 vs stage2 | A 3 / B 20 / tie 177 | A 78 / B 84 / tie 38 | A 68 / B 88 / tie 44 | A 75 / B 105 / tie 20 | A 79 / B 105 / tie 16 |
| stage2 vs dpo | A 8 / B 18 / tie 174 | A 72 / B 99 / tie 29 | A 75 / B 86 / tie 39 | A 82 / B 107 / tie 11 | A 81 / B 109 / tie 10 |
| stage2 vs orpo | A 7 / B 5 / tie 188 | A 77 / B 73 / tie 50 | A 69 / B 73 / tie 58 | A 93 / B 77 / tie 30 | A 96 / B 77 / tie 27 |
| dpo vs orpo | A 13 / B 7 / tie 180 | A 97 / B 75 / tie 28 | A 91 / B 71 / tie 38 | A 105 / B 80 / tie 15 | A 107 / B 80 / tie 13 |

## Quick Takeaways

- **stage1 vs stage2**: overall preference = `A 79 / B 105 / tie 16`; current winner: **B**.
- **stage2 vs dpo**: overall preference = `A 81 / B 109 / tie 10`; current winner: **B**.
- **stage2 vs orpo**: overall preference = `A 96 / B 77 / tie 27`; current winner: **A**.
- **dpo vs orpo**: overall preference = `A 107 / B 80 / tie 13`; current winner: **A**.

