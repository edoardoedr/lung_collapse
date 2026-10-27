# Fit analysis

68 fits, 15 patients. Shape errors measured by this script against the collapsed lung surface, free points only (see the docstring).


## Per fit

| case | run | side | K | collapse | fit coll. | ASSD | fit→tgt | tgt→fit | p95 | rot ° | mean p/E | ap | si | ml | R² | failed | min | stop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| karl01_noreg | fit_torch | left | 40 | 25% | 25% | 2.13 | 3.10 | 1.17 | 7.77 | 4.7 | 0.85 | 0.50 | -1.02 | -1.15 | 0.11 | 0 | 0.3 | levels exhausted |
| karl01_noreg_wall | fit_torch | left | 40 | 25% | 25% | 2.17 | 3.17 | 1.18 | 7.79 | 5.1 | 0.83 | 0.04 | -1.04 | -1.99 | 0.22 | 3 | 4.7 | levels exhausted |
| karl01 | fit_gradient | left | 3 | 26% | 27% | 3.88 | 5.37 | 2.39 | 11.43 | 1.9 | 0.63 | 0.43 | -0.33 | -0.00 | 1.00 | 0 | 0.3 | levels exhausted |
| karl01 | fit_torch | left | 25 | 26% | 26% | 2.05 | 2.73 | 1.37 | 8.07 | 2.6 | 0.53 | 0.39 | -0.42 | -0.18 | 0.12 | 0 | 1.9 | target error reached |
| karl01_wall | fit_torch | left | 25 | 26% | 25% | 1.97 | 2.62 | 1.32 | 7.81 | 2.1 | 0.47 | 0.36 | -0.40 | -0.21 | 0.08 | 0 | 4.0 | target error reached |
| karl02_noreg | fit_torch | right | 40 | 53% | 53% | 2.81 | 2.59 | 3.03 | 7.72 | 8.4 | 1.11 | -0.16 | -0.45 | -0.58 | 0.14 | 0 | 0.3 | target error reached |
| karl02_noreg_wall | fit_torch | right | 40 | 53% | 53% | 2.78 | 2.55 | 3.01 | 7.58 | 8.4 | 1.10 | -0.06 | -0.38 | -0.58 | 0.16 | 0 | 0.5 | target error reached |
| karl02 | fit_gradient | right | 3 | 54% | 51% | 8.04 | 8.42 | 7.65 | 17.15 | 17.0 | 0.88 | 0.17 | -0.02 | 0.00 | 1.00 | 0 | 0.5 | levels exhausted |
| karl02 | fit_torch | right | 40 | 54% | 52% | 2.58 | 2.42 | 2.74 | 7.33 | 14.4 | 0.95 | 0.32 | -0.07 | -0.30 | 0.12 | 0 | 0.5 | target error reached |
| karl02_wall | fit_torch | right | 40 | 54% | 51% | 2.67 | 2.60 | 2.74 | 7.68 | 14.8 | 0.93 | 0.35 | -0.02 | -0.32 | 0.14 | 17 | 40.7 | target error reached |
| karl04_noreg | fit_torch | right | 40 | 51% | 51% | 2.82 | 3.15 | 2.48 | 9.03 | 12.1 | 1.06 | 0.29 | -0.63 | -0.52 | 0.08 | 2 | 1.0 | levels exhausted |
| karl04_noreg_wall | fit_torch | right | 40 | 51% | 51% | 2.84 | 3.18 | 2.49 | 9.17 | 11.8 | 1.05 | 0.30 | -0.62 | -0.53 | 0.08 | 1 | 2.3 | levels exhausted |
| karl04 | fit_gradient | right | 1 | 51% | 45% | 15.17 | 18.71 | 11.63 | 57.90 | 2.6 | 0.78 | -0.00 | 0.00 | -0.00 | – | 0 | 0.5 | levels exhausted |
| karl04 | fit_torch | right | 40 | 51% | 48% | 2.91 | 3.20 | 2.61 | 9.09 | 17.4 | 0.91 | 0.41 | -0.39 | -0.58 | 0.20 | 0 | 2.1 | levels exhausted |
| karl04_wall | fit_torch | right | 25 | 51% | 48% | 3.63 | 3.93 | 3.33 | 11.03 | 16.7 | 0.86 | 0.52 | -0.26 | -0.33 | 0.20 | 19 | 48.3 | levels exhausted |
| karl06_noreg | fit_torch | right | 40 | 23% | 23% | 2.09 | 2.68 | 1.50 | 6.97 | 2.9 | 0.89 | -0.20 | 0.57 | -0.33 | 0.06 | 0 | 0.3 | levels exhausted |
| karl06_noreg_wall | fit_torch | right | 40 | 23% | 23% | 2.09 | 2.68 | 1.50 | 6.97 | 2.9 | 0.89 | -0.20 | 0.57 | -0.33 | 0.06 | 0 | 0.3 | levels exhausted |
| karl06 | fit_gradient | right | 3 | 26% | 27% | 4.38 | 5.26 | 3.50 | 12.92 | 2.5 | 0.54 | 0.29 | -0.24 | 0.00 | 1.00 | 0 | 0.5 | levels exhausted |
| karl06 | fit_torch | right | 14 | 26% | 25% | 2.20 | 2.59 | 1.80 | 6.98 | 5.9 | 0.47 | 0.28 | -0.06 | -0.09 | 0.20 | 0 | 1.3 | target error reached |
| karl06_wall | fit_torch | right | 14 | 26% | 24% | 2.17 | 2.59 | 1.75 | 7.27 | 5.8 | 0.45 | 0.28 | -0.05 | -0.10 | 0.20 | 0 | 2.3 | target error reached |
| karl08_noreg | fit_torch | right | 40 | 28% | 28% | 2.32 | 2.78 | 1.87 | 6.82 | 4.3 | 0.91 | 0.48 | -0.34 | -0.59 | 0.14 | 0 | 0.4 | levels exhausted |
| karl08_noreg_wall | fit_torch | right | 40 | 28% | 28% | 2.32 | 2.78 | 1.87 | 6.82 | 4.3 | 0.91 | 0.48 | -0.34 | -0.59 | 0.14 | 0 | 0.4 | levels exhausted |
| karl08 | fit_gradient | right | 3 | 30% | 28% | 6.11 | 7.14 | 5.08 | 18.02 | 7.5 | 0.53 | 0.25 | -0.30 | -0.00 | 1.00 | 0 | 0.4 | levels exhausted |
| karl08 | fit_torch | right | 40 | 30% | 29% | 2.06 | 2.30 | 1.82 | 6.34 | 4.6 | 0.55 | 0.26 | -0.24 | -0.09 | 0.12 | 0 | 2.4 | target error reached |
| karl08_wall | fit_torch | right | 40 | 30% | 29% | 1.99 | 2.23 | 1.75 | 6.41 | 4.3 | 0.55 | 0.25 | -0.24 | -0.09 | 0.12 | 0 | 5.5 | target error reached |
| karl10_noreg | fit_torch | left | 25 | 56% | 56% | 2.19 | 2.50 | 1.88 | 6.83 | 3.8 | 1.52 | 0.09 | -1.22 | -1.84 | 0.47 | 0 | 0.2 | target error reached |
| karl10_noreg_wall | fit_torch | left | 25 | 56% | 56% | 2.19 | 2.50 | 1.88 | 6.83 | 3.8 | 1.52 | 0.09 | -1.22 | -1.84 | 0.47 | 0 | 0.2 | target error reached |
| karl10 | fit_gradient | left | 3 | 58% | 57% | 6.49 | 6.90 | 6.09 | 19.20 | 12.3 | 1.03 | 0.32 | 0.16 | -0.00 | 1.00 | 0 | 0.4 | levels exhausted |
| karl10 | fit_torch | left | 40 | 58% | 57% | 2.29 | 2.53 | 2.05 | 7.57 | 4.7 | 1.05 | 0.63 | 0.34 | 0.31 | 0.20 | 0 | 2.3 | target error reached |
| karl10_wall | fit_torch | left | 8 | 58% | 55% | 4.29 | 4.94 | 3.64 | 12.41 | 7.3 | 0.95 | 0.57 | 0.52 | 0.45 | 0.52 | 15 | 33.1 | fit plateau (2 levels < 2% gain) |
| karl17_noreg | fit_torch | left | 40 | 24% | 23% | 2.21 | 3.22 | 1.20 | 7.72 | 3.5 | 1.22 | 1.95 | -1.60 | 0.61 | 0.27 | 0 | 0.3 | levels exhausted |
| karl17_noreg_wall | fit_torch | left | 40 | 24% | 23% | 2.21 | 3.22 | 1.20 | 7.72 | 3.5 | 1.22 | 1.95 | -1.60 | 0.61 | 0.27 | 0 | 0.3 | levels exhausted |
| karl17 | fit_gradient | left | 3 | 24% | 26% | 5.35 | 7.19 | 3.51 | 17.42 | 6.9 | 0.60 | 0.39 | -0.37 | 0.00 | 1.00 | 0 | 0.5 | levels exhausted |
| karl17 | fit_torch | left | 25 | 24% | 23% | 2.10 | 2.83 | 1.37 | 7.76 | 6.4 | 0.44 | 0.57 | -0.30 | 0.17 | 0.23 | 0 | 1.5 | target error reached |
| karl17_wall | fit_torch | left | 8 | 24% | 23% | 3.15 | 4.29 | 2.01 | 9.44 | 6.3 | 0.46 | 0.42 | -0.23 | 0.25 | 0.23 | 11 | 28.0 | fit plateau (2 levels < 2% gain) |
| karl18_noreg | fit_torch | left | 8 | 31% | 30% | 1.88 | 2.51 | 1.26 | 5.68 | 0.8 | 0.77 | 0.50 | -0.08 | 0.01 | 0.33 | 0 | 0.1 | target error reached |
| karl18_noreg_wall | fit_torch | left | 8 | 31% | 30% | 1.88 | 2.51 | 1.26 | 5.68 | 0.8 | 0.77 | 0.50 | -0.08 | 0.01 | 0.33 | 0 | 0.1 | target error reached |
| karl18 | fit_torch | left | 8 | 32% | 31% | 1.40 | 1.62 | 1.18 | 4.15 | 0.8 | 0.59 | 0.35 | 0.06 | 0.35 | 0.27 | 0 | 1.0 | target error reached |
| karl18_wall | fit_torch | left | 4 | 32% | 31% | 1.79 | 2.14 | 1.45 | 5.11 | 1.2 | 0.56 | 0.30 | 0.13 | 0.46 | 0.38 | 0 | 0.9 | target error reached |
| karl20_noreg | fit_torch | left | 40 | 58% | 58% | 2.24 | 2.53 | 1.96 | 7.39 | 12.3 | 1.61 | 1.46 | -1.48 | 0.07 | 0.26 | 0 | 0.4 | target error reached |
| karl20_noreg_wall | fit_torch | left | 40 | 58% | 58% | 2.24 | 2.53 | 1.96 | 7.39 | 12.3 | 1.61 | 1.46 | -1.48 | 0.07 | 0.26 | 0 | 0.4 | target error reached |
| karl20 | fit_torch | left | 25 | 55% | 51% | 2.23 | 2.64 | 1.83 | 7.49 | 13.7 | 0.98 | 1.52 | -0.47 | 0.75 | 0.33 | 0 | 1.4 | target error reached |
| karl20_wall | fit_torch | left | 25 | 55% | 51% | 2.25 | 2.65 | 1.85 | 7.04 | 13.8 | 0.96 | 1.48 | -0.41 | 0.78 | 0.32 | 0 | 3.3 | target error reached |
| karl23_noreg | fit_torch | left | 40 | 51% | 50% | 2.80 | 4.27 | 1.34 | 10.10 | 4.9 | 1.87 | 2.33 | 0.17 | -0.03 | 0.20 | 0 | 0.6 | levels exhausted |
| karl23_noreg_wall | fit_torch | left | 40 | 51% | 50% | 3.15 | 4.84 | 1.47 | 11.49 | 6.9 | 1.77 | 2.37 | -0.10 | 0.29 | 0.20 | 0 | 0.6 | levels exhausted |
| karl23 | fit_torch | left | 25 | 37% | 34% | 1.83 | 2.03 | 1.62 | 5.36 | 14.1 | 0.71 | 0.81 | -0.16 | -0.34 | 0.18 | 0 | 2.5 | target error reached |
| karl23_wall | fit_torch | left | 25 | 37% | 33% | 1.86 | 2.09 | 1.63 | 5.49 | 14.1 | 0.71 | 0.77 | -0.15 | -0.36 | 0.18 | 0 | 4.0 | target error reached |
| karl24_noreg | fit_torch | left | 40 | 24% | 24% | 1.90 | 2.66 | 1.15 | 6.75 | 3.1 | 0.91 | 1.44 | -1.23 | -0.00 | 0.13 | 0 | 0.3 | levels exhausted |
| karl24_noreg_wall | fit_torch | left | 40 | 24% | 24% | 1.90 | 2.66 | 1.15 | 6.74 | 3.1 | 0.91 | 1.43 | -1.22 | 0.00 | 0.13 | 0 | 0.5 | levels exhausted |
| karl24 | fit_torch | left | 25 | 26% | 25% | 1.96 | 2.56 | 1.36 | 9.32 | 7.9 | 0.42 | 0.07 | -0.36 | 0.04 | 0.10 | 0 | 3.0 | target error reached |
| karl24_wall | fit_torch | left | 25 | 26% | 25% | 1.94 | 2.54 | 1.34 | 9.04 | 7.8 | 0.42 | 0.05 | -0.35 | 0.04 | 0.10 | 0 | 4.6 | target error reached |
| patient_1_noreg | fit_torch | left | 40 | 59% | 59% | 4.09 | 5.03 | 3.15 | 12.09 | 7.4 | 1.60 | 0.60 | -0.62 | 0.08 | 0.04 | 1 | 0.5 | levels exhausted |
| patient_1_noreg_wall | fit_torch | left | 40 | 59% | 59% | 4.09 | 5.03 | 3.15 | 12.09 | 7.4 | 1.60 | 0.60 | -0.62 | 0.08 | 0.04 | 1 | 1.3 | levels exhausted |
| patient_1 | fit_torch | left | 40 | 59% | 56% | 2.87 | 2.98 | 2.76 | 8.94 | 24.9 | 0.94 | 0.03 | -0.24 | -0.49 | 0.06 | 0 | 0.9 | levels exhausted |
| patient_1_wall | fit_torch | left | 14 | 59% | 53% | 6.70 | 7.25 | 6.16 | 18.17 | 27.2 | 0.90 | 0.33 | -0.09 | -0.37 | 0.11 | 17 | 34.8 | fit plateau (2 levels < 2% gain) |
| patient_10_noreg | fit_torch | left | 40 | 47% | 46% | 1.93 | 2.33 | 1.54 | 6.71 | 7.4 | 0.99 | 0.59 | -0.28 | -0.33 | 0.14 | 0 | 0.3 | target error reached |
| patient_10_noreg_wall | fit_torch | left | 40 | 47% | 46% | 1.93 | 2.33 | 1.54 | 6.71 | 7.4 | 0.99 | 0.59 | -0.28 | -0.33 | 0.14 | 0 | 0.3 | target error reached |
| patient_10 | fit_gradient | left | 3 | 48% | 48% | 6.96 | 7.87 | 6.04 | 22.24 | 13.0 | 0.85 | 0.21 | -0.02 | -0.00 | 1.00 | 0 | 0.5 | levels exhausted |
| patient_10 | fit_torch | left | 40 | 48% | 47% | 2.14 | 2.38 | 1.89 | 6.74 | 13.0 | 0.80 | 0.64 | -0.20 | -0.18 | 0.17 | 0 | 2.3 | target error reached |
| patient_10_wall | fit_torch | left | 40 | 48% | 47% | 2.15 | 2.40 | 1.90 | 6.72 | 13.0 | 0.80 | 0.64 | -0.20 | -0.18 | 0.17 | 0 | 4.5 | target error reached |
| patient_2_noreg | fit_torch | left | 40 | 86% | 87% | 3.57 | 2.56 | 4.58 | 8.10 | 26.3 | 2.06 | -0.06 | -2.06 | -0.45 | 0.25 | 3 | 1.2 | levels exhausted |
| patient_2_noreg_wall | fit_torch | left | 40 | 86% | 87% | 3.57 | 2.56 | 4.58 | 8.10 | 26.3 | 2.06 | -0.06 | -2.06 | -0.45 | 0.25 | 3 | 4.4 | levels exhausted |
| patient_2 | fit_torch | left | 40 | 87% | 80% | 9.29 | 11.43 | 7.16 | 31.68 | 19.1 | 1.90 | 2.03 | -0.84 | -0.39 | 0.43 | 0 | 3.0 | levels exhausted |
| patient_2_wall | fit_torch | left | 14 | 87% | 78% | 14.61 | 19.53 | 9.68 | 45.24 | 7.9 | 1.48 | 1.07 | -0.69 | 0.19 | 0.41 | 27 | 56.8 | fit plateau (2 levels < 2% gain) |
| patient_3_noreg | fit_torch | left | 40 | 37% | 36% | 2.30 | 3.12 | 1.48 | 8.17 | 5.5 | 1.12 | 1.42 | -1.21 | 0.06 | 0.20 | 1 | 0.8 | levels exhausted |
| patient_3_noreg_wall | fit_torch | left | 40 | 37% | 36% | 2.30 | 3.12 | 1.48 | 8.09 | 5.4 | 1.11 | 1.41 | -1.19 | 0.07 | 0.19 | 2 | 3.9 | levels exhausted |
| patient_3 | fit_torch | left | 40 | 38% | 36% | 2.30 | 2.88 | 1.71 | 10.25 | 8.3 | 0.64 | 0.36 | -0.39 | -0.22 | 0.13 | 0 | 2.2 | target error reached |
| patient_3_wall | fit_torch | left | 4 | 38% | 36% | 4.40 | 5.44 | 3.36 | 15.21 | 6.0 | 0.62 | 0.20 | -0.47 | 0.05 | 0.32 | 2 | 6.5 | fit plateau (2 levels < 2% gain) |


## Per variant (median over patients)

| variant | n | ASSD | fit→tgt | tgt→fit | vol err % | rot ° | ap > 0 | si < 0 | rel ap | rel si | mean p/E vs collapse c | min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| noreg / still 5 / no wall / regions | 15 | 2.24 | 2.68 | 1.54 | 0.8 | 4.9 | 12/15 | 13/15 | 0.59 | -0.59 | 0.44 + 1.83·c (max res 0.50) | 0.3 |
| noreg / still 5 / wall / regions | 15 | 2.24 | 2.68 | 1.54 | 0.8 | 5.4 | 12/15 | 14/15 | 0.53 | -0.59 | 0.43 + 1.82·c (max res 0.41) | 0.5 |
| reg / still 5 / no wall / gradient | 8 | 6.30 | 7.16 | 5.56 | 0.7 | 7.2 | 7/8 | 6/8 | 0.39 | -0.24 | 0.26 + 1.18·c (max res 0.09) | 0.5 |
| reg / still 5 / no wall / regions | 15 | 2.20 | 2.59 | 1.82 | 3.0 | 8.3 | 15/15 | 13/15 | 0.60 | -0.43 | -0.10 + 2.06·c (max res 0.22) | 2.1 |
| reg / still 5 / wall / regions | 15 | 2.25 | 2.62 | 1.90 | 4.2 | 7.8 | 15/15 | 13/15 | 0.61 | -0.31 | 0.04 + 1.61·c (max res 0.10) | 5.5 |


## Paired comparisons (B − A, median over patients; negative error = B better)

| comparison | fixed | n | B better (ASSD) | Δ ASSD | Δ fit→tgt | Δ tgt→fit | Δ rot ° | Δ mean p/E | Δ ap | Δ failed | Δ min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| registration: reg → noreg | still 5, no wall, regions | 15 | 6/15 | 0.01 | 0.17 | -0.17 | -2.9 | 0.42 | 0.11 | 0 | -1.4 |
| registration: reg → noreg | still 5, wall, regions | 15 | 10/15 | -0.08 | -0.07 | -0.25 | -2.9 | 0.49 | -0.02 | 0 | -4.2 |
| wall: False → True | noreg, still 5, regions | 15 | 11/15 | -0.00 | -0.00 | -0.00 | 0.0 | -0.00 | -0.00 | 0 | 0.0 |
| wall: False → True | reg, still 5, regions | 15 | 4/15 | 0.09 | 0.18 | 0.02 | -0.1 | -0.02 | -0.04 | 0 | 3.1 |
| model: regions → gradient | reg, still 5, no wall | 8 | 0/8 | 4.13 | 4.61 | 3.65 | 0.2 | 0.02 | -0.17 | 0 | -1.6 |


## Pressure pattern vs collapse and clamp (Pearson r over patients, per variant)

| variant | n | rel ap ~ collapse | rel si ~ collapse | rel ap ~ clamp | rel si ~ clamp | collapse ~ clamp |
|---|---|---|---|---|---|---|
| noreg / still 5 / no wall / regions | 15 | -0.41 | -0.04 | 0.57 | -0.04 | -0.92 |
| noreg / still 5 / wall / regions | 15 | -0.33 | -0.04 | 0.50 | -0.05 | -0.92 |
| reg / still 5 / no wall / gradient | 8 | -0.86 | 0.98 | 0.86 | -0.81 | -0.84 |
| reg / still 5 / no wall / regions | 15 | 0.10 | 0.30 | 0.38 | -0.30 | -0.78 |
| reg / still 5 / wall / regions | 15 | 0.13 | 0.29 | 0.27 | -0.26 | -0.78 |
