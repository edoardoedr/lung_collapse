# Fit analysis

120 fits, 15 patients. Shape errors measured by this script against the collapsed lung surface, free points only (see the docstring).


## Per fit

| case | run | side | K | collapse | fit coll. | ASSD | fit→tgt | tgt→fit | p95 | rot ° | mean p/E | ap | si | ml | R² | failed | min | stop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| karl01_noreg_hilum | fit_grad_ap | left | 2 | 25% | 33% | 6.38 | 7.30 | 5.46 | 21.43 | 9.2 | 0.58 | 0.14 | -0.00 | -0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl01_noreg_hilum | fit_grad_apsi | left | 3 | 25% | 37% | 6.66 | 6.98 | 6.34 | 16.67 | 10.8 | 0.65 | 0.17 | -0.13 | 0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl01_noreg_hilum | fit_torch | left | 40 | 25% | 28% | 2.48 | 2.64 | 2.33 | 8.20 | 5.1 | 0.43 | 0.18 | -0.25 | -0.04 | 0.05 | 0 | 0.6 | levels exhausted |
| karl01_noreg_hilum_wall | fit_torch | left | 8 | 25% | 32% | 4.72 | 5.50 | 3.94 | 14.21 | 10.5 | 0.54 | 0.13 | -0.14 | -0.05 | 0.39 | 15 | 30.8 | fit plateau (2 levels < 2% gain) |
| karl01_noreg | fit_torch | left | 40 | 25% | 25% | 2.13 | 3.10 | 1.17 | 7.77 | 4.7 | 0.85 | 0.50 | -1.02 | -1.15 | 0.11 | 0 | 0.3 | levels exhausted |
| karl01_noreg_wall | fit_torch | left | 40 | 25% | 25% | 2.17 | 3.17 | 1.18 | 7.79 | 5.1 | 0.83 | 0.04 | -1.04 | -1.99 | 0.22 | 3 | 4.7 | levels exhausted |
| karl01 | fit_gradient | left | 3 | 26% | 27% | 3.88 | 5.37 | 2.39 | 11.43 | 1.9 | 0.63 | 0.43 | -0.33 | -0.00 | 1.00 | 0 | 0.3 | levels exhausted |
| karl01 | fit_torch | left | 25 | 26% | 26% | 2.05 | 2.73 | 1.37 | 8.07 | 2.6 | 0.53 | 0.39 | -0.42 | -0.18 | 0.12 | 0 | 1.9 | target error reached |
| karl01_wall | fit_torch | left | 25 | 26% | 25% | 1.97 | 2.62 | 1.32 | 7.81 | 2.1 | 0.47 | 0.36 | -0.40 | -0.21 | 0.08 | 0 | 4.0 | target error reached |
| karl02_noreg_hilum | fit_grad_ap | right | 2 | 53% | 53% | 8.44 | 8.18 | 8.71 | 18.01 | 18.1 | 0.91 | 0.12 | -0.00 | 0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl02_noreg_hilum | fit_grad_apsi | right | 3 | 53% | 53% | 8.50 | 8.16 | 8.83 | 17.30 | 18.4 | 0.91 | 0.12 | -0.04 | -0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl02_noreg_hilum | fit_torch | right | 40 | 53% | 54% | 3.43 | 2.69 | 4.17 | 7.97 | 16.2 | 0.97 | 0.23 | -0.13 | -0.31 | 0.20 | 0 | 0.6 | levels exhausted |
| karl02_noreg_hilum_wall | fit_torch | right | 4 | 53% | 51% | 6.65 | 6.42 | 6.88 | 15.79 | 12.8 | 0.84 | 0.35 | 0.02 | 0.14 | 0.70 | 12 | 24.0 | fit plateau (2 levels < 2% gain) |
| karl02_noreg | fit_torch | right | 40 | 53% | 53% | 2.81 | 2.59 | 3.03 | 7.72 | 8.4 | 1.11 | -0.16 | -0.45 | -0.58 | 0.14 | 0 | 0.3 | target error reached |
| karl02_noreg_wall | fit_torch | right | 40 | 53% | 53% | 2.78 | 2.55 | 3.01 | 7.58 | 8.4 | 1.10 | -0.06 | -0.38 | -0.58 | 0.16 | 0 | 0.5 | target error reached |
| karl02 | fit_gradient | right | 3 | 54% | 51% | 8.04 | 8.42 | 7.65 | 17.15 | 17.0 | 0.88 | 0.17 | -0.02 | 0.00 | 1.00 | 0 | 0.5 | levels exhausted |
| karl02 | fit_torch | right | 40 | 54% | 52% | 2.58 | 2.42 | 2.74 | 7.33 | 14.4 | 0.95 | 0.32 | -0.07 | -0.30 | 0.12 | 0 | 0.5 | target error reached |
| karl02_wall | fit_torch | right | 40 | 54% | 51% | 2.67 | 2.60 | 2.74 | 7.68 | 14.8 | 0.93 | 0.35 | -0.02 | -0.32 | 0.14 | 17 | 40.7 | target error reached |
| karl04_noreg_hilum | fit_grad_ap | right | 2 | 51% | 53% | 12.97 | 14.54 | 11.40 | 42.18 | 17.5 | 0.90 | 0.10 | -0.00 | 0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl04_noreg_hilum | fit_grad_apsi | right | 3 | 51% | 53% | 12.20 | 13.22 | 11.18 | 32.13 | 21.9 | 0.91 | 0.14 | -0.14 | -0.00 | 1.00 | 0 | 0.3 | levels exhausted |
| karl04_noreg_hilum | fit_torch | right | 40 | 51% | 52% | 3.51 | 3.50 | 3.53 | 9.82 | 31.1 | 0.98 | 0.08 | -0.37 | -0.67 | 0.14 | 0 | 0.8 | levels exhausted |
| karl04_noreg_hilum_wall | fit_torch | right | 4 | 51% | 52% | 8.84 | 9.56 | 8.13 | 31.81 | 21.4 | 0.84 | 0.32 | -0.08 | -0.00 | 0.36 | 12 | 24.2 | fit plateau (2 levels < 2% gain) |
| karl04_noreg | fit_torch | right | 40 | 51% | 51% | 2.82 | 3.15 | 2.48 | 9.03 | 12.1 | 1.06 | 0.29 | -0.63 | -0.52 | 0.08 | 2 | 1.0 | levels exhausted |
| karl04_noreg_wall | fit_torch | right | 40 | 51% | 51% | 2.84 | 3.18 | 2.49 | 9.17 | 11.8 | 1.05 | 0.30 | -0.62 | -0.53 | 0.08 | 1 | 2.3 | levels exhausted |
| karl04 | fit_gradient | right | 1 | 51% | 45% | 15.17 | 18.71 | 11.63 | 57.90 | 2.6 | 0.78 | -0.00 | 0.00 | -0.00 | – | 0 | 0.5 | levels exhausted |
| karl04 | fit_torch | right | 40 | 51% | 48% | 2.91 | 3.20 | 2.61 | 9.09 | 17.4 | 0.91 | 0.41 | -0.39 | -0.58 | 0.20 | 0 | 2.1 | levels exhausted |
| karl04_wall | fit_torch | right | 25 | 51% | 48% | 3.63 | 3.93 | 3.33 | 11.03 | 16.7 | 0.86 | 0.52 | -0.26 | -0.33 | 0.20 | 19 | 48.3 | levels exhausted |
| karl06_noreg_hilum | fit_grad_ap | right | 2 | 23% | 29% | 5.61 | 6.07 | 5.15 | 16.70 | 7.5 | 0.50 | 0.09 | 0.00 | -0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl06_noreg_hilum | fit_grad_apsi | right | 3 | 23% | 28% | 5.20 | 5.60 | 4.80 | 15.52 | 7.4 | 0.48 | 0.08 | -0.04 | -0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl06_noreg_hilum | fit_torch | right | 25 | 23% | 27% | 2.40 | 2.35 | 2.46 | 6.98 | 10.7 | 0.43 | 0.16 | -0.12 | -0.14 | 0.12 | 0 | 0.3 | target error reached |
| karl06_noreg_hilum_wall | fit_torch | right | 25 | 23% | 26% | 2.34 | 2.34 | 2.33 | 6.88 | 11.0 | 0.42 | 0.16 | -0.10 | -0.14 | 0.13 | 0 | 0.5 | target error reached |
| karl06_noreg | fit_torch | right | 40 | 23% | 23% | 2.09 | 2.68 | 1.50 | 6.97 | 2.9 | 0.89 | -0.20 | 0.57 | -0.33 | 0.06 | 0 | 0.3 | levels exhausted |
| karl06_noreg_wall | fit_torch | right | 40 | 23% | 23% | 2.09 | 2.68 | 1.50 | 6.97 | 2.9 | 0.89 | -0.20 | 0.57 | -0.33 | 0.06 | 0 | 0.3 | levels exhausted |
| karl06 | fit_gradient | right | 3 | 26% | 27% | 4.38 | 5.26 | 3.50 | 12.92 | 2.5 | 0.54 | 0.29 | -0.24 | 0.00 | 1.00 | 0 | 0.5 | levels exhausted |
| karl06 | fit_torch | right | 14 | 26% | 25% | 2.20 | 2.59 | 1.80 | 6.98 | 5.9 | 0.47 | 0.28 | -0.06 | -0.09 | 0.20 | 0 | 1.3 | target error reached |
| karl06_wall | fit_torch | right | 14 | 26% | 24% | 2.17 | 2.59 | 1.75 | 7.27 | 5.8 | 0.45 | 0.28 | -0.05 | -0.10 | 0.20 | 0 | 2.3 | target error reached |
| karl08_noreg_hilum | fit_grad_ap | right | 2 | 28% | 30% | 7.53 | 8.22 | 6.84 | 19.31 | 7.2 | 0.52 | 0.07 | 0.00 | -0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl08_noreg_hilum | fit_grad_apsi | right | 3 | 28% | 30% | 6.80 | 7.32 | 6.29 | 16.42 | 8.0 | 0.52 | 0.09 | -0.18 | -0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl08_noreg_hilum | fit_torch | right | 40 | 28% | 30% | 2.34 | 2.20 | 2.49 | 5.47 | 7.7 | 0.53 | 0.22 | -0.16 | -0.12 | 0.07 | 0 | 0.3 | target error reached |
| karl08_noreg_hilum_wall | fit_torch | right | 40 | 28% | 29% | 2.47 | 2.38 | 2.57 | 6.31 | 8.3 | 0.52 | 0.21 | -0.13 | -0.12 | 0.07 | 0 | 0.6 | target error reached |
| karl08_noreg | fit_torch | right | 40 | 28% | 28% | 2.32 | 2.78 | 1.87 | 6.82 | 4.3 | 0.91 | 0.48 | -0.34 | -0.59 | 0.14 | 0 | 0.4 | levels exhausted |
| karl08_noreg_wall | fit_torch | right | 40 | 28% | 28% | 2.32 | 2.78 | 1.87 | 6.82 | 4.3 | 0.91 | 0.48 | -0.34 | -0.59 | 0.14 | 0 | 0.4 | levels exhausted |
| karl08 | fit_gradient | right | 3 | 30% | 28% | 6.11 | 7.14 | 5.08 | 18.02 | 7.5 | 0.53 | 0.25 | -0.30 | -0.00 | 1.00 | 0 | 0.4 | levels exhausted |
| karl08 | fit_torch | right | 40 | 30% | 29% | 2.06 | 2.30 | 1.82 | 6.34 | 4.6 | 0.55 | 0.26 | -0.24 | -0.09 | 0.12 | 0 | 2.4 | target error reached |
| karl08_wall | fit_torch | right | 40 | 30% | 29% | 1.99 | 2.23 | 1.75 | 6.41 | 4.3 | 0.55 | 0.25 | -0.24 | -0.09 | 0.12 | 0 | 5.5 | target error reached |
| karl10_noreg_hilum | fit_grad_ap | left | 2 | 56% | 74% | 12.60 | 6.87 | 18.34 | 19.85 | 25.2 | 1.29 | 0.47 | -0.00 | -0.00 | 1.00 | 0 | 0.3 | levels exhausted |
| karl10_noreg_hilum | fit_grad_apsi | left | 3 | 56% | 63% | 7.37 | 6.60 | 8.15 | 17.42 | 6.0 | 1.11 | -0.05 | 0.39 | 0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl10_noreg_hilum | fit_torch | left | 40 | 56% | 58% | 2.89 | 2.50 | 3.29 | 6.86 | 6.2 | 0.97 | 0.22 | 0.20 | 0.31 | 0.06 | 0 | 0.4 | target error reached |
| karl10_noreg_hilum_wall | fit_torch | left | 14 | 56% | 58% | 4.31 | 3.80 | 4.81 | 9.24 | 9.4 | 0.98 | 0.53 | 0.48 | 0.24 | 0.41 | 12 | 25.0 | fit plateau (2 levels < 2% gain) |
| karl10_noreg | fit_torch | left | 25 | 56% | 56% | 2.19 | 2.50 | 1.88 | 6.83 | 3.8 | 1.52 | 0.09 | -1.22 | -1.84 | 0.47 | 0 | 0.2 | target error reached |
| karl10_noreg_wall | fit_torch | left | 25 | 56% | 56% | 2.19 | 2.50 | 1.88 | 6.83 | 3.8 | 1.52 | 0.09 | -1.22 | -1.84 | 0.47 | 0 | 0.2 | target error reached |
| karl10 | fit_gradient | left | 3 | 58% | 57% | 6.49 | 6.90 | 6.09 | 19.20 | 12.3 | 1.03 | 0.32 | 0.16 | -0.00 | 1.00 | 0 | 0.4 | levels exhausted |
| karl10 | fit_torch | left | 40 | 58% | 57% | 2.29 | 2.53 | 2.05 | 7.57 | 4.7 | 1.05 | 0.63 | 0.34 | 0.31 | 0.20 | 0 | 2.3 | target error reached |
| karl10_wall | fit_torch | left | 8 | 58% | 55% | 4.29 | 4.94 | 3.64 | 12.41 | 7.3 | 0.95 | 0.57 | 0.52 | 0.45 | 0.52 | 15 | 33.1 | fit plateau (2 levels < 2% gain) |
| karl17_noreg_hilum | fit_grad_ap | left | 2 | 24% | 36% | 8.19 | 8.59 | 7.79 | 19.55 | 15.9 | 0.62 | 0.13 | -0.00 | 0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl17_noreg_hilum | fit_grad_apsi | left | 3 | 24% | 33% | 7.70 | 8.50 | 6.91 | 23.62 | 11.6 | 0.56 | 0.09 | -0.06 | 0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl17_noreg_hilum | fit_torch | left | 40 | 24% | 26% | 3.21 | 3.33 | 3.08 | 9.18 | 10.6 | 0.42 | 0.10 | -0.20 | -0.01 | 0.02 | 0 | 0.5 | levels exhausted |
| karl17_noreg_hilum_wall | fit_torch | left | 8 | 24% | 23% | 4.45 | 5.03 | 3.88 | 12.91 | 15.4 | 0.33 | 0.17 | 0.00 | 0.08 | 0.11 | 14 | 29.3 | fit plateau (2 levels < 2% gain) |
| karl17_noreg | fit_torch | left | 40 | 24% | 23% | 2.21 | 3.22 | 1.20 | 7.72 | 3.5 | 1.22 | 1.95 | -1.60 | 0.61 | 0.27 | 0 | 0.3 | levels exhausted |
| karl17_noreg_wall | fit_torch | left | 40 | 24% | 23% | 2.21 | 3.22 | 1.20 | 7.72 | 3.5 | 1.22 | 1.95 | -1.60 | 0.61 | 0.27 | 0 | 0.3 | levels exhausted |
| karl17 | fit_gradient | left | 3 | 24% | 26% | 5.35 | 7.19 | 3.51 | 17.42 | 6.9 | 0.60 | 0.39 | -0.37 | 0.00 | 1.00 | 0 | 0.5 | levels exhausted |
| karl17 | fit_torch | left | 25 | 24% | 23% | 2.10 | 2.83 | 1.37 | 7.76 | 6.4 | 0.44 | 0.57 | -0.30 | 0.17 | 0.23 | 0 | 1.5 | target error reached |
| karl17_wall | fit_torch | left | 8 | 24% | 23% | 3.15 | 4.29 | 2.01 | 9.44 | 6.3 | 0.46 | 0.42 | -0.23 | 0.25 | 0.23 | 11 | 28.0 | fit plateau (2 levels < 2% gain) |
| karl18_noreg_hilum | fit_grad_ap | left | 1 | 31% | 38% | 6.02 | 5.93 | 6.11 | 12.03 | 1.5 | 0.65 | -0.00 | -0.00 | 0.00 | – | 0 | 0.1 | levels exhausted |
| karl18_noreg_hilum | fit_grad_apsi | left | 3 | 31% | 42% | 6.02 | 5.04 | 7.01 | 13.53 | 7.1 | 0.72 | 0.14 | 0.16 | 0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl18_noreg_hilum | fit_torch | left | 14 | 31% | 31% | 2.33 | 2.32 | 2.34 | 5.43 | 0.8 | 0.53 | 0.22 | 0.14 | 0.27 | 0.22 | 0 | 0.2 | target error reached |
| karl18_noreg_hilum_wall | fit_torch | left | 14 | 31% | 30% | 2.16 | 2.22 | 2.10 | 5.32 | 1.4 | 0.50 | 0.17 | 0.16 | 0.30 | 0.22 | 0 | 0.3 | target error reached |
| karl18_noreg | fit_torch | left | 8 | 31% | 30% | 1.88 | 2.51 | 1.26 | 5.68 | 0.8 | 0.77 | 0.50 | -0.08 | 0.01 | 0.33 | 0 | 0.1 | target error reached |
| karl18_noreg_wall | fit_torch | left | 8 | 31% | 30% | 1.88 | 2.51 | 1.26 | 5.68 | 0.8 | 0.77 | 0.50 | -0.08 | 0.01 | 0.33 | 0 | 0.1 | target error reached |
| karl18 | fit_torch | left | 8 | 32% | 31% | 1.40 | 1.62 | 1.18 | 4.15 | 0.8 | 0.59 | 0.35 | 0.06 | 0.35 | 0.27 | 0 | 1.0 | target error reached |
| karl18_wall | fit_torch | left | 4 | 32% | 31% | 1.79 | 2.14 | 1.45 | 5.11 | 1.2 | 0.56 | 0.30 | 0.13 | 0.46 | 0.38 | 0 | 0.9 | target error reached |
| karl20_noreg_hilum | fit_grad_ap | left | 2 | 58% | 77% | 12.60 | 10.86 | 14.33 | 24.57 | 21.4 | 1.40 | 1.02 | -0.00 | -0.00 | 1.00 | 0 | 0.3 | levels exhausted |
| karl20_noreg_hilum | fit_grad_apsi | left | 3 | 58% | 72% | 11.06 | 9.75 | 12.38 | 21.57 | 22.4 | 1.31 | 0.83 | -0.59 | -0.00 | 1.00 | 0 | 0.3 | levels exhausted |
| karl20_noreg_hilum | fit_torch | left | 40 | 58% | 60% | 3.63 | 3.30 | 3.96 | 8.54 | 20.5 | 1.31 | 1.32 | -1.59 | 0.23 | 0.37 | 0 | 0.7 | levels exhausted |
| karl20_noreg_hilum_wall | fit_torch | left | 4 | 58% | 68% | 8.37 | 7.84 | 8.89 | 17.74 | 19.5 | 1.14 | 0.79 | -0.47 | 0.01 | 0.42 | 11 | 21.3 | fit plateau (2 levels < 2% gain) |
| karl20_noreg | fit_torch | left | 40 | 58% | 58% | 2.24 | 2.53 | 1.96 | 7.39 | 12.3 | 1.61 | 1.46 | -1.48 | 0.07 | 0.26 | 0 | 0.4 | target error reached |
| karl20_noreg_wall | fit_torch | left | 40 | 58% | 58% | 2.24 | 2.53 | 1.96 | 7.39 | 12.3 | 1.61 | 1.46 | -1.48 | 0.07 | 0.26 | 0 | 0.4 | target error reached |
| karl20 | fit_torch | left | 25 | 55% | 51% | 2.23 | 2.64 | 1.83 | 7.49 | 13.7 | 0.98 | 1.52 | -0.47 | 0.75 | 0.33 | 0 | 1.4 | target error reached |
| karl20_wall | fit_torch | left | 25 | 55% | 51% | 2.25 | 2.65 | 1.85 | 7.04 | 13.8 | 0.96 | 1.48 | -0.41 | 0.78 | 0.32 | 0 | 3.3 | target error reached |
| karl23_noreg_hilum | fit_grad_ap | left | 2 | 51% | 95% | 28.27 | 12.51 | 44.03 | 22.91 | 56.8 | 3.00 | -3.49 | 0.00 | -0.00 | 1.00 | 0 | 0.3 | levels exhausted |
| karl23_noreg_hilum | fit_grad_apsi | left | 3 | 51% | 94% | 27.57 | 11.72 | 43.41 | 22.69 | 61.4 | 2.92 | -4.00 | -2.71 | -0.00 | 1.00 | 0 | 0.3 | levels exhausted |
| karl23_noreg_hilum | fit_torch | left | 40 | 51% | 93% | 24.46 | 9.02 | 39.90 | 23.65 | 47.3 | 2.32 | -1.68 | 0.39 | 1.34 | 0.27 | 0 | 0.3 | fit plateau (2 levels < 2% gain) |
| karl23_noreg_hilum_wall | fit_torch | left | 40 | 51% | 93% | 24.43 | 8.99 | 39.88 | 23.83 | 47.5 | 2.31 | -1.75 | 0.50 | 1.32 | 0.27 | 0 | 0.4 | fit plateau (2 levels < 2% gain) |
| karl23_noreg | fit_torch | left | 40 | 51% | 50% | 2.80 | 4.27 | 1.34 | 10.10 | 4.9 | 1.87 | 2.33 | 0.17 | -0.03 | 0.20 | 0 | 0.6 | levels exhausted |
| karl23_noreg_wall | fit_torch | left | 40 | 51% | 50% | 3.15 | 4.84 | 1.47 | 11.49 | 6.9 | 1.77 | 2.37 | -0.10 | 0.29 | 0.20 | 0 | 0.6 | levels exhausted |
| karl23 | fit_torch | left | 25 | 37% | 34% | 1.83 | 2.03 | 1.62 | 5.36 | 14.1 | 0.71 | 0.81 | -0.16 | -0.34 | 0.18 | 0 | 2.5 | target error reached |
| karl23_wall | fit_torch | left | 25 | 37% | 33% | 1.86 | 2.09 | 1.63 | 5.49 | 14.1 | 0.71 | 0.77 | -0.15 | -0.36 | 0.18 | 0 | 4.0 | target error reached |
| karl24_noreg_hilum | fit_grad_ap | left | 2 | 24% | 31% | 6.11 | 6.81 | 5.40 | 17.88 | 10.0 | 0.52 | 0.05 | -0.00 | 0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl24_noreg_hilum | fit_grad_apsi | left | 3 | 24% | 28% | 5.20 | 5.89 | 4.52 | 17.18 | 10.7 | 0.47 | 0.03 | -0.07 | 0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl24_noreg_hilum | fit_torch | left | 40 | 24% | 27% | 2.20 | 2.44 | 1.95 | 8.52 | 7.5 | 0.42 | 0.05 | -0.17 | -0.01 | 0.03 | 0 | 0.4 | target error reached |
| karl24_noreg_hilum_wall | fit_torch | left | 4 | 24% | 25% | 4.52 | 5.01 | 4.02 | 13.09 | 14.3 | 0.39 | 0.12 | -0.17 | 0.02 | 0.60 | 2 | 2.6 | fit plateau (2 levels < 2% gain) |
| karl24_noreg | fit_torch | left | 40 | 24% | 24% | 1.90 | 2.66 | 1.15 | 6.75 | 3.1 | 0.91 | 1.44 | -1.23 | -0.00 | 0.13 | 0 | 0.3 | levels exhausted |
| karl24_noreg_wall | fit_torch | left | 40 | 24% | 24% | 1.90 | 2.66 | 1.15 | 6.74 | 3.1 | 0.91 | 1.43 | -1.22 | 0.00 | 0.13 | 0 | 0.5 | levels exhausted |
| karl24 | fit_torch | left | 25 | 26% | 25% | 1.96 | 2.56 | 1.36 | 9.32 | 7.9 | 0.42 | 0.07 | -0.36 | 0.04 | 0.10 | 0 | 3.0 | target error reached |
| karl24_wall | fit_torch | left | 25 | 26% | 25% | 1.94 | 2.54 | 1.34 | 9.04 | 7.8 | 0.42 | 0.05 | -0.35 | 0.04 | 0.10 | 0 | 4.6 | target error reached |
| patient_1_noreg_hilum | fit_torch | left | 40 | 59% | 95% | 25.09 | 7.66 | 42.52 | 21.62 | 23.7 | 2.17 | 0.11 | 2.53 | 2.75 | 0.29 | 1 | 0.7 | levels exhausted |
| patient_1_noreg_hilum_wall | fit_torch | left | 40 | 59% | 95% | 24.32 | 8.34 | 40.29 | 20.91 | 17.5 | 2.27 | -0.39 | 2.06 | 2.28 | 0.27 | 0 | 0.8 | levels exhausted |
| patient_1_noreg | fit_torch | left | 40 | 59% | 59% | 4.09 | 5.03 | 3.15 | 12.09 | 7.4 | 1.60 | 0.60 | -0.62 | 0.08 | 0.04 | 1 | 0.5 | levels exhausted |
| patient_1_noreg_wall | fit_torch | left | 40 | 59% | 59% | 4.09 | 5.03 | 3.15 | 12.09 | 7.4 | 1.60 | 0.60 | -0.62 | 0.08 | 0.04 | 1 | 1.3 | levels exhausted |
| patient_1 | fit_torch | left | 40 | 59% | 56% | 2.87 | 2.98 | 2.76 | 8.94 | 24.9 | 0.94 | 0.03 | -0.24 | -0.49 | 0.06 | 0 | 0.9 | levels exhausted |
| patient_1_wall | fit_torch | left | 14 | 59% | 53% | 6.70 | 7.25 | 6.16 | 18.17 | 27.2 | 0.90 | 0.33 | -0.09 | -0.37 | 0.11 | 17 | 34.8 | fit plateau (2 levels < 2% gain) |
| patient_10_noreg_hilum | fit_torch | left | 40 | 47% | 49% | 2.95 | 3.03 | 2.88 | 8.13 | 19.5 | 0.83 | 0.41 | -0.13 | -0.18 | 0.09 | 0 | 0.4 | levels exhausted |
| patient_10_noreg_hilum_wall | fit_torch | left | 4 | 47% | 49% | 5.99 | 6.17 | 5.81 | 16.49 | 22.2 | 0.79 | 0.38 | 0.01 | -0.02 | 0.56 | 14 | 29.8 | fit plateau (2 levels < 2% gain) |
| patient_10_noreg | fit_torch | left | 40 | 47% | 46% | 1.93 | 2.33 | 1.54 | 6.71 | 7.4 | 0.99 | 0.59 | -0.28 | -0.33 | 0.14 | 0 | 0.3 | target error reached |
| patient_10_noreg_wall | fit_torch | left | 40 | 47% | 46% | 1.93 | 2.33 | 1.54 | 6.71 | 7.4 | 0.99 | 0.59 | -0.28 | -0.33 | 0.14 | 0 | 0.3 | target error reached |
| patient_10 | fit_gradient | left | 3 | 48% | 48% | 6.96 | 7.87 | 6.04 | 22.24 | 13.0 | 0.85 | 0.21 | -0.02 | -0.00 | 1.00 | 0 | 0.5 | levels exhausted |
| patient_10 | fit_torch | left | 40 | 48% | 47% | 2.14 | 2.38 | 1.89 | 6.74 | 13.0 | 0.80 | 0.64 | -0.20 | -0.18 | 0.17 | 0 | 2.3 | target error reached |
| patient_10_wall | fit_torch | left | 40 | 48% | 47% | 2.15 | 2.40 | 1.90 | 6.72 | 13.0 | 0.80 | 0.64 | -0.20 | -0.18 | 0.17 | 0 | 4.5 | target error reached |
| patient_2_noreg_hilum | fit_torch | left | 1 | 86% | 98% | 27.08 | 8.93 | 45.23 | 20.65 | 24.0 | 3.00 | 0.00 | 0.00 | 0.00 | – | 0 | 0.1 | fit plateau (2 levels < 2% gain) |
| patient_2_noreg_hilum_wall | fit_torch | left | 1 | 86% | 98% | 27.08 | 8.93 | 45.23 | 20.65 | 24.0 | 3.00 | 0.00 | 0.00 | 0.00 | – | 0 | 0.1 | fit plateau (2 levels < 2% gain) |
| patient_2_noreg | fit_torch | left | 40 | 86% | 87% | 3.57 | 2.56 | 4.58 | 8.10 | 26.3 | 2.06 | -0.06 | -2.06 | -0.45 | 0.25 | 3 | 1.2 | levels exhausted |
| patient_2_noreg_wall | fit_torch | left | 40 | 86% | 87% | 3.57 | 2.56 | 4.58 | 8.10 | 26.3 | 2.06 | -0.06 | -2.06 | -0.45 | 0.25 | 3 | 4.4 | levels exhausted |
| patient_2 | fit_torch | left | 40 | 87% | 80% | 9.29 | 11.43 | 7.16 | 31.68 | 19.1 | 1.90 | 2.03 | -0.84 | -0.39 | 0.43 | 0 | 3.0 | levels exhausted |
| patient_2_wall | fit_torch | left | 14 | 87% | 78% | 14.61 | 19.53 | 9.68 | 45.24 | 7.9 | 1.48 | 1.07 | -0.69 | 0.19 | 0.41 | 27 | 56.8 | fit plateau (2 levels < 2% gain) |
| patient_3_noreg_hilum | fit_torch | left | 40 | 37% | 39% | 2.88 | 3.14 | 2.62 | 9.69 | 10.0 | 0.61 | 0.40 | -0.37 | 0.18 | 0.10 | 0 | 0.6 | levels exhausted |
| patient_3_noreg_hilum_wall | fit_torch | left | 4 | 37% | 39% | 5.67 | 6.02 | 5.32 | 15.62 | 13.1 | 0.62 | 0.25 | -0.43 | 0.08 | 0.64 | 32 | 64.7 | fit plateau (2 levels < 2% gain) |
| patient_3_noreg | fit_torch | left | 40 | 37% | 36% | 2.30 | 3.12 | 1.48 | 8.17 | 5.5 | 1.12 | 1.42 | -1.21 | 0.06 | 0.20 | 1 | 0.8 | levels exhausted |
| patient_3_noreg_wall | fit_torch | left | 40 | 37% | 36% | 2.30 | 3.12 | 1.48 | 8.09 | 5.4 | 1.11 | 1.41 | -1.19 | 0.07 | 0.19 | 2 | 3.9 | levels exhausted |
| patient_3 | fit_torch | left | 40 | 38% | 36% | 2.30 | 2.88 | 1.71 | 10.25 | 8.3 | 0.64 | 0.36 | -0.39 | -0.22 | 0.13 | 0 | 2.2 | target error reached |
| patient_3_wall | fit_torch | left | 4 | 38% | 36% | 4.40 | 5.44 | 3.36 | 15.21 | 6.0 | 0.62 | 0.20 | -0.47 | 0.05 | 0.32 | 2 | 6.5 | fit plateau (2 levels < 2% gain) |


## Per variant (median over patients)

| variant | n | ASSD | fit→tgt | tgt→fit | vol err % | rot ° | ap > 0 | si < 0 | rel ap | rel si | mean p/E vs collapse c | min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| noreg / hilum only / no wall / gradient ap / one-sided | 11 | 8.19 | 8.18 | 7.79 | -10.3 | 15.9 | 9/11 | 8/11 | 0.14 | -0.00 | -0.22 + 3.12·c (max res 1.62) | 0.2 |
| noreg / hilum only / no wall / gradient ap+si / one-sided | 11 | 7.37 | 7.32 | 7.01 | -11.9 | 10.8 | 9/11 | 9/11 | 0.16 | -0.14 | -0.15 + 2.88·c (max res 1.60) | 0.2 |
| noreg / hilum only / no wall / regions / one-sided | 15 | 2.95 | 3.03 | 3.08 | -3.4 | 10.7 | 14/15 | 10/15 | 0.23 | -0.28 | -0.62 + 3.86·c (max res 0.97) | 0.4 |
| noreg / hilum only / wall / regions / one-sided | 15 | 5.67 | 6.02 | 5.32 | -3.2 | 14.3 | 13/15 | 7/15 | 0.38 | 0.00 | -0.63 + 3.80·c (max res 0.99) | 21.3 |
| noreg / still 5 / no wall / regions / one-sided | 15 | 2.24 | 2.68 | 1.54 | 0.8 | 4.9 | 12/15 | 13/15 | 0.59 | -0.59 | 0.44 + 1.83·c (max res 0.50) | 0.3 |
| noreg / still 5 / wall / regions / one-sided | 15 | 2.24 | 2.68 | 1.54 | 0.8 | 5.4 | 12/15 | 14/15 | 0.53 | -0.59 | 0.43 + 1.82·c (max res 0.41) | 0.5 |
| reg / still 5 / no wall / gradient ap+si / nodes | 8 | 6.30 | 7.16 | 5.56 | 0.7 | 7.2 | 7/8 | 6/8 | 0.39 | -0.24 | 0.26 + 1.18·c (max res 0.09) | 0.5 |
| reg / still 5 / no wall / regions / nodes | 15 | 2.20 | 2.59 | 1.82 | 3.0 | 8.3 | 15/15 | 13/15 | 0.60 | -0.43 | -0.10 + 2.06·c (max res 0.22) | 2.1 |
| reg / still 5 / wall / regions / nodes | 15 | 2.25 | 2.62 | 1.90 | 4.2 | 7.8 | 15/15 | 13/15 | 0.61 | -0.31 | 0.04 + 1.61·c (max res 0.10) | 5.5 |


## Paired comparisons (B − A, median over patients; negative error = B better)

| comparison | fixed | n | B better (ASSD) | Δ ASSD | Δ fit→tgt | Δ tgt→fit | Δ rot ° | Δ mean p/E | Δ ap | Δ failed | Δ min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clamp: still 5 → hilum only | noreg, no wall, regions, one-sided | 15 | 0/15 | 0.70 | 0.10 | 1.16 | 7.0 | -0.30 | -0.26 | 0 | 0.1 |
| clamp: still 5 → hilum only | noreg, wall, regions, one-sided | 15 | 0/15 | 3.37 | 2.90 | 3.84 | 7.7 | -0.29 | -0.27 | 11 | 20.9 |
| wall: False → True | noreg, hilum only, regions, one-sided | 15 | 4/15 | 1.41 | 1.70 | 1.52 | 0.6 | -0.02 | -0.01 | 11 | 20.6 |
| wall: False → True | noreg, still 5, regions, one-sided | 15 | 11/15 | -0.00 | -0.00 | -0.00 | 0.0 | -0.00 | -0.00 | 0 | 0.0 |
| wall: False → True | reg, still 5, regions, nodes | 15 | 4/15 | 0.09 | 0.18 | 0.02 | -0.1 | -0.02 | -0.04 | 0 | 3.1 |
| model: regions → gradient ap+si | noreg, hilum only, no wall, one-sided | 33 | 8/33 | 3.69 | 3.61 | 3.52 | 1.0 | 0.05 | -0.05 | 0 | -0.1 |
| model: regions → gradient ap+si | reg, still 5, no wall, nodes | 8 | 0/8 | 4.13 | 4.61 | 3.65 | 0.2 | 0.02 | -0.17 | 0 | -1.6 |
| model: regions → gradient ap | noreg, hilum only, no wall, one-sided | 33 | 8/33 | 3.69 | 3.61 | 3.52 | 1.0 | 0.05 | -0.05 | 0 | -0.1 |
| model: regions → gradient ap | reg, still 5, no wall, nodes | 8 | 0/8 | 4.13 | 4.61 | 3.65 | 0.2 | 0.02 | -0.17 | 0 | -1.6 |
| model: gradient ap → gradient ap+si | noreg, hilum only, no wall, one-sided | 33 | 8/33 | 3.69 | 3.61 | 3.52 | 1.0 | 0.05 | -0.05 | 0 | -0.1 |
| model: gradient ap → gradient ap+si | reg, still 5, no wall, nodes | 8 | 0/8 | 4.13 | 4.61 | 3.65 | 0.2 | 0.02 | -0.17 | 0 | -1.6 |


## Pressure pattern vs collapse and clamp (Pearson r over patients, per variant)

| variant | n | rel ap ~ collapse | rel si ~ collapse | rel ap ~ clamp | rel si ~ clamp | collapse ~ clamp |
|---|---|---|---|---|---|---|
| noreg / hilum only / no wall / gradient ap / one-sided | 11 | -0.03 | -0.40 | -0.09 | 0.43 | -0.19 |
| noreg / hilum only / no wall / gradient ap+si / one-sided | 11 | -0.19 | -0.16 | 0.02 | 0.03 | -0.19 |
| noreg / hilum only / no wall / regions / one-sided | 15 | -0.22 | 0.28 | 0.22 | -0.13 | -0.47 |
| noreg / hilum only / wall / regions / one-sided | 15 | -0.27 | 0.37 | 0.18 | -0.13 | -0.47 |
| noreg / still 5 / no wall / regions / one-sided | 15 | -0.41 | -0.04 | 0.57 | -0.04 | -0.92 |
| noreg / still 5 / wall / regions / one-sided | 15 | -0.33 | -0.04 | 0.50 | -0.05 | -0.92 |
| reg / still 5 / no wall / gradient ap+si / nodes | 8 | -0.86 | 0.98 | 0.86 | -0.81 | -0.84 |
| reg / still 5 / no wall / regions / nodes | 15 | 0.10 | 0.30 | 0.38 | -0.30 | -0.78 |
| reg / still 5 / wall / regions / nodes | 15 | 0.13 | 0.29 | 0.27 | -0.26 | -0.78 |
