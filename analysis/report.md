# Fit analysis

193 fits, 15 patients. Shape errors measured by this script against the collapsed lung surface, free points only (see the docstring).


## Per fit

| case | run | side | K | collapse | fit coll. | ASSD | fit→tgt | tgt→fit | p95 | rot ° | mean p/E | ap | si | ml | R² | failed | min | stop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| karl01_noreg_hilum | fit_grad_ap | left | 2 | 25% | 33% | 6.38 | 7.30 | 5.46 | 21.43 | 9.2 | 0.58 | 0.14 | -0.00 | -0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl01_noreg_hilum | fit_grad_ap_sym | left | 2 | 25% | 30% | 6.16 | 7.40 | 4.92 | 24.15 | 7.8 | 0.52 | 0.11 | 0.00 | 0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl01_noreg_hilum | fit_grad_apsi | left | 3 | 25% | 37% | 6.66 | 6.98 | 6.34 | 16.67 | 10.8 | 0.65 | 0.17 | -0.13 | 0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl01_noreg_hilum | fit_grad_apsi_sym | left | 3 | 25% | 30% | 6.11 | 7.24 | 4.99 | 21.22 | 7.8 | 0.52 | 0.11 | -0.06 | -0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl01_noreg_hilum | fit_torch | left | 40 | 25% | 28% | 2.48 | 2.64 | 2.33 | 8.20 | 5.1 | 0.43 | 0.18 | -0.25 | -0.04 | 0.05 | 0 | 0.6 | levels exhausted |
| karl01_noreg_hilum | fit_sym | left | 40 | 25% | 25% | 2.40 | 2.87 | 1.94 | 8.95 | 4.3 | 0.36 | 0.11 | -0.12 | -0.04 | 0.03 | 0 | 0.4 | target error reached |
| karl01_noreg_hilum | fit_sym_k0.001 | left | 40 | 25% | 26% | 2.30 | 2.68 | 1.92 | 8.45 | 3.3 | 0.38 | 0.21 | -0.17 | -0.01 | 0.08 | 0 | 7.7 | target error reached |
| karl01_noreg_hilum | fit_sym_k0.003 | left | 25 | 25% | 27% | 3.30 | 4.07 | 2.54 | 13.17 | 2.8 | 0.44 | 0.31 | -0.18 | -0.14 | 0.23 | 16 | 32.7 | fit plateau (2 levels < 2% gain) |
| karl01_noreg_hilum_wall | fit_grad_ap_sym | left | 2 | 25% | 28% | 5.76 | 6.96 | 4.56 | 18.88 | 13.3 | 0.48 | 0.26 | -0.00 | -0.00 | 1.00 | 0 | 0.5 | levels exhausted |
| karl01_noreg_hilum_wall | fit_torch | left | 8 | 25% | 32% | 4.72 | 5.50 | 3.94 | 14.21 | 10.5 | 0.54 | 0.13 | -0.14 | -0.05 | 0.39 | 15 | 30.8 | fit plateau (2 levels < 2% gain) |
| karl01_noreg_hilum_wall | fit_sym | left | 8 | 25% | 24% | 3.55 | 4.37 | 2.73 | 12.81 | 6.9 | 0.40 | 0.18 | -0.20 | -0.16 | 0.23 | 14 | 41.1 | fit plateau (2 levels < 2% gain) |
| karl01_noreg_hilum_wall | fit_sym_k0.001 | left | 8 | 25% | 27% | 3.70 | 4.58 | 2.82 | 13.72 | 5.0 | 0.45 | 0.34 | -0.22 | -0.16 | 0.29 | 12 | 27.8 | fit plateau (2 levels < 2% gain) |
| karl01_noreg_hilum_wall | fit_sym_k0.003 | left | 4 | 25% | 26% | 6.27 | 8.45 | 4.08 | 33.42 | 1.3 | 0.46 | 0.15 | -0.03 | 0.03 | 0.34 | 14 | 32.7 | fit plateau (2 levels < 2% gain) |
| karl02_noreg_hilum | fit_grad_ap | right | 2 | 53% | 53% | 8.44 | 8.18 | 8.71 | 18.01 | 18.1 | 0.91 | 0.12 | -0.00 | 0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl02_noreg_hilum | fit_grad_ap_sym | right | 2 | 53% | 49% | 8.31 | 8.37 | 8.25 | 18.67 | 17.0 | 0.84 | 0.10 | -0.00 | 0.00 | 1.00 | 0 | 0.3 | levels exhausted |
| karl02_noreg_hilum | fit_grad_apsi | right | 3 | 53% | 53% | 8.50 | 8.16 | 8.83 | 17.30 | 18.4 | 0.91 | 0.12 | -0.04 | -0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl02_noreg_hilum | fit_grad_apsi_sym | right | 3 | 53% | 49% | 8.30 | 8.35 | 8.25 | 18.29 | 17.1 | 0.84 | 0.10 | -0.01 | -0.00 | 1.00 | 0 | 0.3 | levels exhausted |
| karl02_noreg_hilum | fit_torch | right | 40 | 53% | 54% | 3.43 | 2.69 | 4.17 | 7.97 | 16.2 | 0.97 | 0.23 | -0.13 | -0.31 | 0.20 | 0 | 0.6 | levels exhausted |
| karl02_noreg_hilum | fit_sym | right | 40 | 53% | 52% | 3.20 | 2.85 | 3.55 | 8.31 | 13.1 | 0.93 | 0.28 | -0.08 | -0.25 | 0.12 | 0 | 0.5 | levels exhausted |
| karl02_noreg_hilum | fit_sym_k0.001 | right | 40 | 53% | 52% | 2.90 | 2.59 | 3.21 | 7.85 | 8.2 | 0.95 | 0.49 | -0.13 | -0.23 | 0.24 | 0 | 9.9 | levels exhausted |
| karl02_noreg_hilum | fit_sym_k0.003 | right | 40 | 53% | 52% | 2.72 | 2.48 | 2.96 | 6.93 | 5.7 | 1.01 | 0.75 | -0.21 | -0.34 | 0.22 | 0 | 7.9 | levels exhausted |
| karl02_noreg_hilum_wall | fit_grad_ap_sym | right | 2 | 53% | 49% | 5.30 | 5.15 | 5.45 | 13.92 | 15.0 | 0.83 | 0.32 | -0.00 | -0.00 | 1.00 | 4 | 8.4 | levels exhausted |
| karl02_noreg_hilum_wall | fit_torch | right | 4 | 53% | 51% | 6.65 | 6.42 | 6.88 | 15.79 | 12.8 | 0.84 | 0.35 | 0.02 | 0.14 | 0.70 | 12 | 24.0 | fit plateau (2 levels < 2% gain) |
| karl02_noreg_hilum_wall | fit_sym | right | 4 | 53% | 49% | 6.57 | 6.55 | 6.59 | 15.98 | 11.7 | 0.79 | 0.31 | 0.02 | 0.12 | 0.75 | 12 | 26.1 | fit plateau (2 levels < 2% gain) |
| karl02_noreg_hilum_wall | fit_sym_k0.001 | right | 4 | 53% | 49% | 6.49 | 6.70 | 6.28 | 17.70 | 5.6 | 0.81 | 0.54 | -0.11 | 0.22 | 0.73 | 11 | 22.2 | fit plateau (2 levels < 2% gain) |
| karl02_noreg_hilum_wall | fit_sym_k0.003 | right | 40 | 53% | 52% | 2.71 | 2.47 | 2.95 | 6.95 | 5.8 | 1.01 | 0.75 | -0.22 | -0.35 | 0.21 | 11 | 32.1 | levels exhausted |
| karl04_noreg_hilum | fit_grad_ap | right | 2 | 51% | 53% | 12.97 | 14.54 | 11.40 | 42.18 | 17.5 | 0.90 | 0.10 | -0.00 | 0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl04_noreg_hilum | fit_grad_ap_sym | right | 2 | 51% | 46% | 12.22 | 14.65 | 9.78 | 46.20 | 16.0 | 0.78 | 0.08 | -0.00 | 0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl04_noreg_hilum | fit_grad_apsi | right | 3 | 51% | 53% | 12.20 | 13.22 | 11.18 | 32.13 | 21.9 | 0.91 | 0.14 | -0.14 | -0.00 | 1.00 | 0 | 0.3 | levels exhausted |
| karl04_noreg_hilum | fit_grad_apsi_sym | right | 3 | 51% | 45% | 11.40 | 13.58 | 9.21 | 42.08 | 16.4 | 0.77 | 0.09 | -0.08 | -0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl04_noreg_hilum | fit_torch | right | 40 | 51% | 52% | 3.51 | 3.50 | 3.53 | 9.82 | 31.1 | 0.98 | 0.08 | -0.37 | -0.67 | 0.14 | 0 | 0.8 | levels exhausted |
| karl04_noreg_hilum | fit_sym | right | 40 | 51% | 51% | 3.90 | 4.06 | 3.73 | 11.24 | 31.8 | 0.96 | 0.03 | -0.34 | -0.62 | 0.15 | 0 | 0.7 | levels exhausted |
| karl04_noreg_hilum | fit_sym_k0.001 | right | 40 | 51% | 50% | 3.21 | 3.39 | 3.03 | 10.36 | 16.9 | 0.95 | 0.37 | -0.48 | -0.53 | 0.15 | 0 | 10.4 | levels exhausted |
| karl04_noreg_hilum | fit_sym_k0.003 | right | 40 | 51% | 50% | 3.26 | 3.52 | 2.99 | 10.42 | 9.5 | 1.04 | 0.77 | -0.80 | -0.49 | 0.26 | 0 | 9.2 | levels exhausted |
| karl04_noreg_hilum_wall | fit_grad_ap_sym | right | 2 | 51% | 46% | 8.62 | 10.44 | 6.80 | 35.63 | 16.3 | 0.75 | 0.45 | -0.00 | 0.00 | 1.00 | 4 | 9.4 | levels exhausted |
| karl04_noreg_hilum_wall | fit_torch | right | 4 | 51% | 52% | 8.84 | 9.56 | 8.13 | 31.81 | 21.4 | 0.84 | 0.32 | -0.08 | -0.00 | 0.36 | 12 | 24.2 | fit plateau (2 levels < 2% gain) |
| karl04_noreg_hilum_wall | fit_sym | right | 4 | 51% | 45% | 8.33 | 9.83 | 6.82 | 30.53 | 18.3 | 0.71 | 0.34 | -0.00 | 0.04 | 0.44 | 14 | 30.5 | fit plateau (2 levels < 2% gain) |
| karl04_noreg_hilum_wall | fit_sym_k0.001 | right | 14 | 51% | 48% | 5.58 | 6.07 | 5.10 | 18.35 | 15.8 | 0.87 | 0.85 | -0.49 | -0.47 | 0.39 | 22 | 44.4 | fit plateau (2 levels < 2% gain) |
| karl04_noreg_hilum_wall | fit_sym_k0.001_fix | right | 14 | 51% | 47% | 5.88 | 6.67 | 5.09 | 20.90 | 14.9 | 0.82 | 0.87 | -0.32 | -0.30 | 0.40 | 11 | 4.3 | signal 2 |
| karl04_noreg_hilum_wall | fit_sym_k0.001_q2 | right | 4 | 51% | 45% | 6.79 | 8.04 | 5.53 | 24.51 | 13.2 | 0.75 | 0.70 | -0.25 | -0.09 | 0.32 | 20 | 40.5 | fit plateau (2 levels < 2% gain) |
| karl04_noreg_hilum_wall | fit_sym_k0.001_t20 | right | 14 | 51% | 48% | 5.58 | 6.07 | 5.10 | 18.35 | 15.8 | 0.87 | 0.85 | -0.49 | -0.47 | 0.39 | 22 | 7.9 | fit plateau (2 levels < 2% gain) |
| karl04_noreg_hilum_wall | fit_sym_k0.001_t300 | right | 14 | 51% | 48% | 5.58 | 6.07 | 5.10 | 18.35 | 15.8 | 0.87 | 0.85 | -0.49 | -0.47 | 0.39 | 22 | 97.2 | fit plateau (2 levels < 2% gain) |
| karl04_noreg_hilum_wall | fit_sym_k0.003 | right | 14 | 51% | 48% | 5.03 | 5.57 | 4.50 | 15.13 | 9.1 | 0.93 | 1.16 | -0.84 | -0.56 | 0.39 | 10 | 30.2 | fit plateau (2 levels < 2% gain) |
| karl06_noreg_hilum | fit_grad_ap | right | 2 | 23% | 29% | 5.61 | 6.07 | 5.15 | 16.70 | 7.5 | 0.50 | 0.09 | 0.00 | -0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl06_noreg_hilum | fit_grad_ap_sym | right | 2 | 23% | 25% | 5.29 | 6.00 | 4.57 | 18.57 | 6.2 | 0.42 | 0.07 | 0.00 | -0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl06_noreg_hilum | fit_grad_apsi | right | 3 | 23% | 28% | 5.20 | 5.60 | 4.80 | 15.52 | 7.4 | 0.48 | 0.08 | -0.04 | -0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl06_noreg_hilum | fit_grad_apsi_sym | right | 3 | 23% | 24% | 4.91 | 5.53 | 4.29 | 16.72 | 6.6 | 0.41 | 0.07 | -0.04 | -0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl06_noreg_hilum | fit_torch | right | 25 | 23% | 27% | 2.40 | 2.35 | 2.46 | 6.98 | 10.7 | 0.43 | 0.16 | -0.12 | -0.14 | 0.12 | 0 | 0.3 | target error reached |
| karl06_noreg_hilum | fit_sym | right | 25 | 23% | 24% | 2.31 | 2.48 | 2.14 | 6.70 | 10.8 | 0.38 | 0.16 | -0.11 | -0.14 | 0.17 | 0 | 0.3 | target error reached |
| karl06_noreg_hilum | fit_sym_k0.001 | right | 14 | 23% | 24% | 2.37 | 2.58 | 2.16 | 8.62 | 6.8 | 0.40 | 0.29 | -0.07 | -0.15 | 0.40 | 0 | 4.2 | target error reached |
| karl06_noreg_hilum | fit_sym_k0.003 | right | 14 | 23% | 24% | 2.30 | 2.52 | 2.08 | 8.90 | 4.3 | 0.43 | 0.47 | -0.14 | -0.23 | 0.45 | 0 | 3.5 | target error reached |
| karl06_noreg_hilum_wall | fit_grad_ap_sym | right | 2 | 23% | 21% | 4.95 | 5.61 | 4.28 | 16.28 | 10.1 | 0.35 | 0.21 | -0.00 | -0.00 | 1.00 | 0 | 0.5 | levels exhausted |
| karl06_noreg_hilum_wall | fit_torch | right | 25 | 23% | 26% | 2.34 | 2.34 | 2.33 | 6.88 | 11.0 | 0.42 | 0.16 | -0.10 | -0.14 | 0.13 | 0 | 0.5 | target error reached |
| karl06_noreg_hilum_wall | fit_sym | right | 25 | 23% | 23% | 2.33 | 2.51 | 2.15 | 6.85 | 10.9 | 0.38 | 0.16 | -0.10 | -0.14 | 0.16 | 0 | 3.5 | target error reached |
| karl06_noreg_hilum_wall | fit_sym_k0.001 | right | 14 | 23% | 23% | 2.36 | 2.57 | 2.15 | 8.04 | 6.9 | 0.40 | 0.29 | -0.07 | -0.15 | 0.37 | 0 | 0.4 | target error reached |
| karl06_noreg_hilum_wall | fit_sym_k0.003 | right | 14 | 23% | 24% | 2.27 | 2.50 | 2.04 | 9.20 | 4.3 | 0.42 | 0.47 | -0.13 | -0.22 | 0.42 | 0 | 7.9 | target error reached |
| karl08_noreg_hilum | fit_grad_ap | right | 2 | 28% | 30% | 7.53 | 8.22 | 6.84 | 19.31 | 7.2 | 0.52 | 0.07 | 0.00 | -0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl08_noreg_hilum | fit_grad_ap_sym | right | 2 | 28% | 26% | 7.27 | 8.16 | 6.39 | 20.90 | 7.0 | 0.46 | 0.07 | 0.00 | -0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl08_noreg_hilum | fit_grad_apsi | right | 3 | 28% | 30% | 6.80 | 7.32 | 6.29 | 16.42 | 8.0 | 0.52 | 0.09 | -0.18 | -0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl08_noreg_hilum | fit_grad_apsi_sym | right | 3 | 28% | 26% | 6.57 | 7.27 | 5.87 | 17.62 | 7.4 | 0.46 | 0.08 | -0.15 | -0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl08_noreg_hilum | fit_torch | right | 40 | 28% | 30% | 2.34 | 2.20 | 2.49 | 5.47 | 7.7 | 0.53 | 0.22 | -0.16 | -0.12 | 0.07 | 0 | 0.3 | target error reached |
| karl08_noreg_hilum | fit_sym | right | 40 | 28% | 28% | 2.25 | 2.32 | 2.18 | 6.70 | 7.8 | 0.50 | 0.22 | -0.14 | -0.11 | 0.08 | 0 | 0.4 | target error reached |
| karl08_noreg_hilum | fit_sym_k0.001 | right | 40 | 28% | 28% | 2.17 | 2.25 | 2.09 | 6.88 | 5.2 | 0.52 | 0.37 | -0.23 | -0.07 | 0.18 | 0 | 6.4 | target error reached |
| karl08_noreg_hilum | fit_sym_k0.003 | right | 25 | 28% | 28% | 2.45 | 2.58 | 2.33 | 7.93 | 3.3 | 0.55 | 0.55 | -0.37 | -0.03 | 0.28 | 0 | 5.3 | target error reached |
| karl08_noreg_hilum_wall | fit_grad_ap_sym | right | 2 | 28% | 25% | 6.76 | 7.54 | 5.98 | 20.02 | 10.6 | 0.44 | 0.18 | 0.00 | -0.00 | 1.00 | 0 | 0.4 | levels exhausted |
| karl08_noreg_hilum_wall | fit_torch | right | 40 | 28% | 29% | 2.47 | 2.38 | 2.57 | 6.31 | 8.3 | 0.52 | 0.21 | -0.13 | -0.12 | 0.07 | 0 | 0.6 | target error reached |
| karl08_noreg_hilum_wall | fit_sym | right | 40 | 28% | 27% | 2.23 | 2.32 | 2.13 | 6.94 | 7.6 | 0.48 | 0.20 | -0.15 | -0.07 | 0.06 | 0 | 4.8 | target error reached |
| karl08_noreg_hilum_wall | fit_sym_k0.001 | right | 25 | 28% | 27% | 2.46 | 2.56 | 2.35 | 7.71 | 5.0 | 0.50 | 0.35 | -0.22 | -0.04 | 0.17 | 0 | 0.5 | target error reached |
| karl08_noreg_hilum_wall | fit_sym_k0.003 | right | 25 | 28% | 27% | 2.44 | 2.57 | 2.31 | 7.78 | 3.1 | 0.54 | 0.53 | -0.37 | -0.01 | 0.27 | 0 | 8.2 | target error reached |
| karl10_noreg_hilum | fit_grad_ap | left | 2 | 56% | 74% | 12.60 | 6.87 | 18.34 | 19.85 | 25.2 | 1.29 | 0.47 | -0.00 | -0.00 | 1.00 | 0 | 0.3 | levels exhausted |
| karl10_noreg_hilum | fit_grad_ap_sym | left | 2 | 56% | 54% | 10.24 | 10.55 | 9.93 | 22.24 | 3.2 | 0.93 | 0.01 | -0.00 | 0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl10_noreg_hilum | fit_grad_apsi | left | 3 | 56% | 63% | 7.37 | 6.60 | 8.15 | 17.42 | 6.0 | 1.11 | -0.05 | 0.39 | 0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl10_noreg_hilum | fit_grad_apsi_sym | left | 3 | 56% | 56% | 7.04 | 6.98 | 7.09 | 17.43 | 6.5 | 0.99 | -0.02 | 0.30 | 0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl10_noreg_hilum | fit_torch | left | 40 | 56% | 58% | 2.89 | 2.50 | 3.29 | 6.86 | 6.2 | 0.97 | 0.22 | 0.20 | 0.31 | 0.06 | 0 | 0.4 | target error reached |
| karl10_noreg_hilum | fit_sym | left | 40 | 56% | 55% | 2.75 | 2.71 | 2.78 | 7.82 | 6.0 | 0.90 | 0.16 | 0.09 | 0.41 | 0.04 | 0 | 0.7 | levels exhausted |
| karl10_noreg_hilum | fit_sym_k0.001 | left | 40 | 56% | 56% | 2.55 | 2.44 | 2.65 | 6.74 | 5.1 | 1.06 | 0.54 | 0.27 | 0.23 | 0.09 | 0 | 9.2 | levels exhausted |
| karl10_noreg_hilum | fit_sym_k0.003 | left | 40 | 56% | 56% | 2.49 | 2.39 | 2.59 | 6.67 | 4.7 | 1.07 | 0.72 | 0.41 | 0.22 | 0.14 | 0 | 7.0 | target error reached |
| karl10_noreg_hilum_wall | fit_grad_ap_sym | left | 2 | 56% | 54% | 10.31 | 10.59 | 10.03 | 22.29 | 4.9 | 0.93 | 0.02 | -0.00 | 0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl10_noreg_hilum_wall | fit_torch | left | 14 | 56% | 58% | 4.31 | 3.80 | 4.81 | 9.24 | 9.4 | 0.98 | 0.53 | 0.48 | 0.24 | 0.41 | 12 | 25.0 | fit plateau (2 levels < 2% gain) |
| karl10_noreg_hilum_wall | fit_sym | left | 8 | 56% | 52% | 4.54 | 4.65 | 4.42 | 13.44 | 7.9 | 0.86 | 0.41 | 0.40 | 0.32 | 0.59 | 11 | 29.2 | fit plateau (2 levels < 2% gain) |
| karl10_noreg_hilum_wall | fit_sym_k0.001 | left | 14 | 56% | 54% | 3.75 | 3.80 | 3.70 | 10.16 | 6.5 | 0.93 | 0.61 | 0.54 | 0.27 | 0.43 | 12 | 24.9 | fit plateau (2 levels < 2% gain) |
| karl10_noreg_hilum_wall | fit_sym_k0.003 | left | 14 | 56% | 54% | 3.53 | 3.59 | 3.47 | 8.40 | 4.9 | 0.99 | 0.83 | 0.57 | 0.31 | 0.40 | 11 | 31.5 | fit plateau (2 levels < 2% gain) |
| karl17_noreg_hilum | fit_grad_ap | left | 2 | 24% | 36% | 8.19 | 8.59 | 7.79 | 19.55 | 15.9 | 0.62 | 0.13 | -0.00 | 0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl17_noreg_hilum | fit_grad_ap_sym | left | 2 | 24% | 28% | 7.27 | 8.44 | 6.10 | 27.54 | 12.8 | 0.47 | 0.09 | -0.00 | 0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl17_noreg_hilum | fit_grad_apsi | left | 3 | 24% | 33% | 7.70 | 8.50 | 6.91 | 23.62 | 11.6 | 0.56 | 0.09 | -0.06 | 0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl17_noreg_hilum | fit_grad_apsi_sym | left | 3 | 24% | 28% | 7.15 | 8.31 | 6.00 | 27.13 | 12.0 | 0.47 | 0.09 | -0.04 | 0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl17_noreg_hilum | fit_torch | left | 40 | 24% | 26% | 3.21 | 3.33 | 3.08 | 9.18 | 10.6 | 0.42 | 0.10 | -0.20 | -0.01 | 0.02 | 0 | 0.5 | levels exhausted |
| karl17_noreg_hilum | fit_sym | left | 40 | 24% | 24% | 2.99 | 3.32 | 2.66 | 10.14 | 11.0 | 0.37 | 0.11 | -0.17 | 0.01 | 0.02 | 0 | 0.6 | levels exhausted |
| karl17_noreg_hilum | fit_sym_k0.001 | left | 40 | 24% | 24% | 2.52 | 2.80 | 2.24 | 8.46 | 9.0 | 0.40 | 0.39 | -0.29 | 0.07 | 0.10 | 0 | 8.9 | target error reached |
| karl17_noreg_hilum | fit_sym_k0.003 | left | 40 | 24% | 23% | 2.35 | 2.77 | 1.94 | 8.16 | 6.4 | 0.44 | 0.72 | -0.44 | 0.17 | 0.21 | 0 | 6.1 | target error reached |
| karl17_noreg_hilum_wall | fit_grad_ap_sym | left | 2 | 24% | 24% | 6.10 | 7.20 | 4.99 | 23.97 | 14.0 | 0.40 | 0.27 | -0.00 | 0.00 | 1.00 | 3 | 7.0 | levels exhausted |
| karl17_noreg_hilum_wall | fit_torch | left | 8 | 24% | 23% | 4.45 | 5.03 | 3.88 | 12.91 | 15.4 | 0.33 | 0.17 | 0.00 | 0.08 | 0.11 | 14 | 29.3 | fit plateau (2 levels < 2% gain) |
| karl17_noreg_hilum_wall | fit_sym | left | 4 | 24% | 21% | 4.81 | 5.57 | 4.05 | 15.63 | 14.5 | 0.31 | 0.12 | -0.03 | 0.05 | 0.09 | 15 | 39.3 | fit plateau (2 levels < 2% gain) |
| karl17_noreg_hilum_wall | fit_sym_k0.001 | left | 14 | 24% | 23% | 3.52 | 4.02 | 3.01 | 10.89 | 10.5 | 0.37 | 0.51 | 0.01 | 0.20 | 0.30 | 12 | 25.4 | fit plateau (2 levels < 2% gain) |
| karl17_noreg_hilum_wall | fit_sym_k0.003 | left | 25 | 24% | 21% | 2.47 | 3.05 | 1.90 | 9.44 | 5.5 | 0.40 | 0.71 | -0.37 | 0.17 | 0.25 | 0 | 9.1 | target error reached |
| karl18_noreg_hilum | fit_grad_ap | left | 1 | 31% | 38% | 6.02 | 5.93 | 6.11 | 12.03 | 1.5 | 0.65 | -0.00 | -0.00 | 0.00 | – | 0 | 0.1 | levels exhausted |
| karl18_noreg_hilum | fit_grad_ap_sym | left | 1 | 31% | 29% | 5.75 | 6.10 | 5.40 | 15.29 | 1.1 | 0.50 | -0.00 | -0.00 | -0.00 | – | 0 | 0.1 | levels exhausted |
| karl18_noreg_hilum | fit_grad_apsi | left | 3 | 31% | 42% | 6.02 | 5.04 | 7.01 | 13.53 | 7.1 | 0.72 | 0.14 | 0.16 | 0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl18_noreg_hilum | fit_grad_apsi_sym | left | 1 | 31% | 29% | 5.75 | 6.10 | 5.40 | 15.29 | 1.1 | 0.50 | -0.00 | -0.00 | -0.00 | – | 0 | 0.2 | levels exhausted |
| karl18_noreg_hilum | fit_torch | left | 14 | 31% | 31% | 2.33 | 2.32 | 2.34 | 5.43 | 0.8 | 0.53 | 0.22 | 0.14 | 0.27 | 0.22 | 0 | 0.2 | target error reached |
| karl18_noreg_hilum | fit_sym | left | 14 | 31% | 29% | 2.42 | 2.52 | 2.32 | 5.83 | 1.2 | 0.49 | 0.19 | 0.12 | 0.26 | 0.20 | 0 | 0.2 | target error reached |
| karl18_noreg_hilum | fit_sym_k0.001 | left | 14 | 31% | 29% | 2.35 | 2.44 | 2.26 | 5.78 | 0.6 | 0.49 | 0.23 | 0.10 | 0.28 | 0.25 | 0 | 2.9 | target error reached |
| karl18_noreg_hilum | fit_sym_k0.003 | left | 14 | 31% | 29% | 2.30 | 2.38 | 2.22 | 5.87 | 0.6 | 0.50 | 0.29 | 0.08 | 0.31 | 0.31 | 0 | 2.5 | target error reached |
| karl18_noreg_hilum_wall | fit_grad_ap_sym | left | 1 | 31% | 29% | 5.75 | 6.10 | 5.40 | 15.29 | 1.1 | 0.50 | -0.00 | -0.00 | -0.00 | – | 0 | 0.1 | levels exhausted |
| karl18_noreg_hilum_wall | fit_torch | left | 14 | 31% | 30% | 2.16 | 2.22 | 2.10 | 5.32 | 1.4 | 0.50 | 0.17 | 0.16 | 0.30 | 0.22 | 0 | 0.3 | target error reached |
| karl18_noreg_hilum_wall | fit_sym | left | 14 | 31% | 28% | 2.25 | 2.41 | 2.09 | 5.41 | 1.9 | 0.47 | 0.17 | 0.15 | 0.30 | 0.23 | 0 | 1.9 | target error reached |
| karl18_noreg_hilum_wall | fit_sym_k0.001 | left | 8 | 31% | 28% | 2.42 | 2.61 | 2.24 | 5.57 | 2.0 | 0.46 | 0.20 | 0.14 | 0.34 | 0.33 | 0 | 0.2 | target error reached |
| karl18_noreg_hilum_wall | fit_sym_k0.003 | left | 8 | 31% | 28% | 2.39 | 2.57 | 2.20 | 5.56 | 1.9 | 0.46 | 0.24 | 0.13 | 0.39 | 0.43 | 0 | 3.7 | target error reached |
| karl20_noreg_hilum | fit_grad_ap | left | 2 | 58% | 77% | 12.60 | 10.86 | 14.33 | 24.57 | 21.4 | 1.40 | 1.02 | -0.00 | -0.00 | 1.00 | 0 | 0.3 | levels exhausted |
| karl20_noreg_hilum | fit_grad_ap_sym | left | 2 | 58% | 69% | 12.00 | 12.09 | 11.92 | 27.97 | 23.5 | 1.20 | 0.69 | -0.00 | -0.00 | 1.00 | 0 | 0.3 | levels exhausted |
| karl20_noreg_hilum | fit_grad_apsi | left | 3 | 58% | 72% | 11.06 | 9.75 | 12.38 | 21.57 | 22.4 | 1.31 | 0.83 | -0.59 | -0.00 | 1.00 | 0 | 0.3 | levels exhausted |
| karl20_noreg_hilum | fit_grad_apsi_sym | left | 3 | 58% | 66% | 10.52 | 10.34 | 10.70 | 23.27 | 22.2 | 1.16 | 0.61 | -0.45 | -0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl20_noreg_hilum | fit_torch | left | 40 | 58% | 60% | 3.63 | 3.30 | 3.96 | 8.54 | 20.5 | 1.31 | 1.32 | -1.59 | 0.23 | 0.37 | 0 | 0.7 | levels exhausted |
| karl20_noreg_hilum | fit_sym | left | 40 | 58% | 55% | 3.94 | 4.39 | 3.49 | 10.51 | 20.3 | 1.06 | 1.02 | -0.56 | 0.81 | 0.20 | 0 | 0.6 | levels exhausted |
| karl20_noreg_hilum | fit_sym_k0.001 | left | 40 | 58% | 55% | 3.64 | 4.25 | 3.04 | 11.54 | 19.6 | 0.97 | 1.41 | -0.30 | 0.35 | 0.16 | 0 | 9.3 | levels exhausted |
| karl20_noreg_hilum | fit_sym_k0.003 | left | 40 | 58% | 56% | 3.18 | 3.83 | 2.53 | 10.91 | 13.1 | 1.11 | 1.99 | -0.28 | 0.47 | 0.27 | 3 | 16.7 | levels exhausted |
| karl20_noreg_hilum_wall | fit_grad_ap_sym | left | 2 | 58% | 64% | 9.80 | 10.78 | 8.83 | 29.43 | 20.7 | 1.10 | 0.79 | -0.00 | -0.00 | 1.00 | 8 | 16.6 | levels exhausted |
| karl20_noreg_hilum_wall | fit_torch | left | 4 | 58% | 68% | 8.37 | 7.84 | 8.89 | 17.74 | 19.5 | 1.14 | 0.79 | -0.47 | 0.01 | 0.42 | 11 | 21.3 | fit plateau (2 levels < 2% gain) |
| karl20_noreg_hilum_wall | fit_sym | left | 4 | 58% | 58% | 7.35 | 7.91 | 6.79 | 20.90 | 21.4 | 0.91 | 0.57 | -0.16 | 0.08 | 0.54 | 11 | 24.8 | fit plateau (2 levels < 2% gain) |
| karl20_noreg_hilum_wall | fit_sym_k0.001 | left | 8 | 58% | 57% | 6.02 | 6.55 | 5.49 | 15.94 | 14.6 | 1.03 | 1.26 | -0.82 | 0.15 | 0.49 | 13 | 26.6 | fit plateau (2 levels < 2% gain) |
| karl20_noreg_hilum_wall | fit_sym_k0.003 | left | 4 | 58% | 37% | 11.56 | 17.33 | 5.79 | 52.72 | 5.3 | 0.64 | 0.84 | -0.56 | -0.06 | 0.45 | 15 | 36.2 | fit plateau (2 levels < 2% gain) |
| karl23_noreg_hilum | fit_grad_ap | left | 2 | 51% | 95% | 28.27 | 12.51 | 44.03 | 22.91 | 56.8 | 3.00 | -3.49 | 0.00 | -0.00 | 1.00 | 0 | 0.3 | levels exhausted |
| karl23_noreg_hilum | fit_grad_ap_sym | left | 2 | 51% | 62% | 16.38 | 18.56 | 14.20 | 49.58 | 23.5 | 1.13 | 0.67 | -0.00 | 0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl23_noreg_hilum | fit_grad_apsi | left | 3 | 51% | 94% | 27.57 | 11.72 | 43.41 | 22.69 | 61.4 | 2.92 | -4.00 | -2.71 | -0.00 | 1.00 | 0 | 0.3 | levels exhausted |
| karl23_noreg_hilum | fit_grad_apsi_sym | left | 3 | 51% | 61% | 16.31 | 17.92 | 14.70 | 45.95 | 27.6 | 1.11 | 0.75 | 0.16 | 0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl23_noreg_hilum | fit_torch | left | 40 | 51% | 93% | 24.46 | 9.02 | 39.90 | 23.65 | 47.3 | 2.32 | -1.68 | 0.39 | 1.34 | 0.27 | 0 | 0.3 | fit plateau (2 levels < 2% gain) |
| karl23_noreg_hilum | fit_sym | left | 40 | 51% | 51% | 3.98 | 4.20 | 3.76 | 12.33 | 72.9 | 0.99 | 0.11 | -0.65 | -1.09 | 0.13 | 0 | 1.0 | levels exhausted |
| karl23_noreg_hilum | fit_sym_k0.001 | left | 40 | 51% | 47% | 3.04 | 3.46 | 2.61 | 9.11 | 41.5 | 1.08 | 1.73 | -0.54 | -1.05 | 0.40 | 0 | 10.7 | levels exhausted |
| karl23_noreg_hilum | fit_sym_k0.003 | left | 14 | 51% | 46% | 6.20 | 7.98 | 4.42 | 21.88 | 11.2 | 0.89 | 1.67 | 0.22 | 0.12 | 0.24 | 32 | 36.9 | fit plateau (2 levels < 2% gain) |
| karl23_noreg_hilum_wall | fit_grad_ap_sym | left | 2 | 51% | 61% | 14.24 | 16.81 | 11.67 | 45.62 | 22.5 | 1.10 | 1.09 | 0.00 | 0.00 | 1.00 | 8 | 15.6 | levels exhausted |
| karl23_noreg_hilum_wall | fit_torch | left | 40 | 51% | 93% | 24.43 | 8.99 | 39.88 | 23.83 | 47.5 | 2.31 | -1.75 | 0.50 | 1.32 | 0.27 | 0 | 0.4 | fit plateau (2 levels < 2% gain) |
| karl23_noreg_hilum_wall | fit_sym | left | 14 | 51% | 30% | 13.14 | 20.42 | 5.87 | 64.52 | 17.2 | 0.51 | 0.35 | -0.07 | -0.13 | 0.53 | 26 | 55.8 | fit plateau (2 levels < 2% gain) |
| karl23_noreg_hilum_wall | fit_sym_k0.001 | left | 4 | 51% | 27% | 11.89 | 18.59 | 5.19 | 56.28 | 13.3 | 0.49 | 0.85 | -0.06 | -0.24 | 0.47 | 17 | 35.6 | fit plateau (2 levels < 2% gain) |
| karl23_noreg_hilum_wall | fit_sym_k0.003 | left | 4 | 51% | 25% | 14.54 | 24.36 | 4.72 | 73.06 | 4.8 | 0.49 | 1.03 | 0.10 | -0.21 | 0.42 | 17 | 40.2 | fit plateau (2 levels < 2% gain) |
| karl24_noreg_hilum | fit_grad_ap | left | 2 | 24% | 31% | 6.11 | 6.81 | 5.40 | 17.88 | 10.0 | 0.52 | 0.05 | -0.00 | 0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl24_noreg_hilum | fit_grad_ap_sym | left | 2 | 24% | 26% | 5.70 | 6.81 | 4.59 | 22.78 | 7.5 | 0.44 | 0.03 | -0.00 | 0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl24_noreg_hilum | fit_grad_apsi | left | 3 | 24% | 28% | 5.20 | 5.89 | 4.52 | 17.18 | 10.7 | 0.47 | 0.03 | -0.07 | 0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl24_noreg_hilum | fit_grad_apsi_sym | left | 3 | 24% | 25% | 5.00 | 5.96 | 4.04 | 20.09 | 8.3 | 0.43 | 0.03 | -0.04 | 0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl24_noreg_hilum | fit_torch | left | 40 | 24% | 27% | 2.20 | 2.44 | 1.95 | 8.52 | 7.5 | 0.42 | 0.05 | -0.17 | -0.01 | 0.03 | 0 | 0.4 | target error reached |
| karl24_noreg_hilum | fit_sym | left | 25 | 24% | 25% | 2.45 | 2.93 | 1.97 | 9.94 | 9.0 | 0.40 | 0.02 | -0.18 | 0.01 | 0.07 | 0 | 0.4 | target error reached |
| karl24_noreg_hilum | fit_sym_k0.001 | left | 25 | 24% | 25% | 2.33 | 2.79 | 1.87 | 9.56 | 7.1 | 0.42 | 0.12 | -0.28 | 0.05 | 0.15 | 0 | 4.6 | target error reached |
| karl24_noreg_hilum | fit_sym_k0.003 | left | 25 | 24% | 26% | 2.24 | 2.62 | 1.86 | 8.98 | 5.8 | 0.47 | 0.31 | -0.47 | 0.09 | 0.32 | 0 | 4.5 | target error reached |
| karl24_noreg_hilum_wall | fit_grad_ap_sym | left | 2 | 24% | 26% | 5.60 | 6.68 | 4.53 | 22.15 | 7.9 | 0.44 | 0.03 | -0.00 | 0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| karl24_noreg_hilum_wall | fit_torch | left | 4 | 24% | 25% | 4.52 | 5.01 | 4.02 | 13.09 | 14.3 | 0.39 | 0.12 | -0.17 | 0.02 | 0.60 | 2 | 2.6 | fit plateau (2 levels < 2% gain) |
| karl24_noreg_hilum_wall | fit_sym | left | 4 | 24% | 20% | 4.24 | 5.01 | 3.47 | 14.34 | 13.4 | 0.31 | 0.12 | -0.14 | 0.02 | 0.64 | 10 | 23.1 | fit plateau (2 levels < 2% gain) |
| karl24_noreg_hilum_wall | fit_sym_k0.001 | left | 25 | 24% | 25% | 2.43 | 2.83 | 2.03 | 9.85 | 5.8 | 0.41 | 0.10 | -0.27 | 0.05 | 0.14 | 0 | 0.8 | target error reached |
| karl24_noreg_hilum_wall | fit_sym_k0.003 | left | 25 | 24% | 25% | 2.29 | 2.67 | 1.91 | 9.03 | 5.6 | 0.46 | 0.29 | -0.48 | 0.10 | 0.32 | 0 | 9.0 | target error reached |
| patient_1_noreg_hilum | fit_grad_ap | left | 2 | 59% | 97% | 29.74 | 13.89 | 45.58 | 24.29 | 17.9 | 3.00 | -0.70 | -0.00 | -0.00 | 1.00 | 0 | 0.3 | levels exhausted |
| patient_1_noreg_hilum | fit_grad_ap_sym | left | 2 | 59% | 67% | 13.79 | 14.26 | 13.32 | 35.23 | 23.2 | 1.16 | 0.34 | 0.00 | -0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| patient_1_noreg_hilum | fit_grad_apsi | left | 3 | 59% | 97% | 29.42 | 13.60 | 45.24 | 24.85 | 21.1 | 3.00 | -1.11 | 2.66 | -0.00 | 1.00 | 0 | 0.4 | levels exhausted |
| patient_1_noreg_hilum | fit_grad_apsi_sym | left | 3 | 59% | 61% | 12.48 | 13.95 | 11.00 | 37.71 | 20.0 | 1.07 | 0.23 | -0.23 | -0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| patient_1_noreg_hilum | fit_torch | left | 40 | 59% | 95% | 25.09 | 7.66 | 42.52 | 21.62 | 23.7 | 2.17 | 0.11 | 2.53 | 2.75 | 0.29 | 1 | 0.7 | levels exhausted |
| patient_1_noreg_hilum | fit_sym | left | 40 | 59% | 57% | 2.93 | 2.99 | 2.87 | 8.11 | 34.2 | 0.99 | -0.13 | -0.53 | -0.39 | 0.06 | 0 | 0.6 | levels exhausted |
| patient_1_noreg_hilum | fit_sym_k0.001 | left | 40 | 59% | 56% | 4.57 | 4.94 | 4.21 | 13.30 | 20.4 | 0.87 | 0.07 | -0.19 | 0.09 | 0.01 | 20 | 25.8 | fit plateau (2 levels < 2% gain) |
| patient_1_noreg_hilum | fit_sym_k0.003 | left | 40 | 59% | 56% | 3.78 | 4.27 | 3.29 | 12.58 | 18.5 | 1.08 | 0.43 | -0.77 | -0.02 | 0.06 | 10 | 26.9 | levels exhausted |
| patient_1_noreg_hilum_wall | fit_grad_ap_sym | left | 2 | 59% | 63% | 12.58 | 13.54 | 11.62 | 33.78 | 23.9 | 1.08 | 0.41 | 0.00 | -0.00 | 1.00 | 5 | 11.3 | levels exhausted |
| patient_1_noreg_hilum_wall | fit_torch | left | 40 | 59% | 95% | 24.32 | 8.34 | 40.29 | 20.91 | 17.5 | 2.27 | -0.39 | 2.06 | 2.28 | 0.27 | 0 | 0.8 | levels exhausted |
| patient_1_noreg_hilum_wall | fit_sym | left | 4 | 59% | 56% | 11.37 | 13.84 | 8.91 | 41.79 | 18.7 | 0.92 | 0.10 | 0.06 | -0.05 | 0.21 | 11 | 24.3 | fit plateau (2 levels < 2% gain) |
| patient_1_noreg_hilum_wall | fit_sym_k0.001 | left | 4 | 59% | 49% | 7.52 | 8.95 | 6.09 | 24.62 | 19.4 | 0.77 | 0.50 | -0.12 | -0.23 | 0.18 | 13 | 26.6 | fit plateau (2 levels < 2% gain) |
| patient_1_noreg_hilum_wall | fit_sym_k0.003 | left | 14 | 59% | 29% | 12.57 | 19.07 | 6.08 | 57.08 | 6.0 | 0.51 | 0.29 | -0.18 | -0.23 | 0.26 | 46 | 88.9 | fit plateau (2 levels < 2% gain) |
| patient_10_noreg_hilum | fit_grad_ap | left | 2 | 47% | 53% | 8.79 | 8.96 | 8.61 | 19.66 | 20.6 | 0.90 | 0.25 | -0.00 | -0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| patient_10_noreg_hilum | fit_grad_ap_sym | left | 2 | 47% | 48% | 8.56 | 9.18 | 7.93 | 21.21 | 18.9 | 0.82 | 0.21 | 0.00 | -0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| patient_10_noreg_hilum | fit_grad_apsi | left | 3 | 47% | 52% | 8.06 | 8.21 | 7.91 | 20.14 | 20.1 | 0.88 | 0.25 | -0.09 | -0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| patient_10_noreg_hilum | fit_grad_apsi_sym | left | 3 | 47% | 48% | 7.84 | 8.42 | 7.26 | 23.17 | 17.7 | 0.81 | 0.20 | -0.08 | 0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| patient_10_noreg_hilum | fit_torch | left | 40 | 47% | 49% | 2.95 | 3.03 | 2.88 | 8.13 | 19.5 | 0.83 | 0.41 | -0.13 | -0.18 | 0.09 | 0 | 0.4 | levels exhausted |
| patient_10_noreg_hilum | fit_sym | left | 40 | 47% | 47% | 2.92 | 3.12 | 2.73 | 7.85 | 20.1 | 0.80 | 0.41 | -0.10 | -0.13 | 0.07 | 0 | 0.7 | levels exhausted |
| patient_10_noreg_hilum | fit_sym_k0.001 | left | 40 | 47% | 47% | 2.50 | 2.65 | 2.36 | 7.39 | 14.2 | 0.83 | 0.66 | -0.23 | -0.22 | 0.16 | 0 | 9.4 | target error reached |
| patient_10_noreg_hilum | fit_sym_k0.003 | left | 40 | 47% | 47% | 2.49 | 2.67 | 2.31 | 8.05 | 9.4 | 0.92 | 1.08 | -0.51 | -0.38 | 0.24 | 0 | 7.1 | target error reached |
| patient_10_noreg_hilum_wall | fit_grad_ap_sym | left | 2 | 47% | 46% | 7.51 | 8.34 | 6.68 | 22.40 | 16.8 | 0.78 | 0.28 | -0.00 | -0.00 | 1.00 | 6 | 12.5 | levels exhausted |
| patient_10_noreg_hilum_wall | fit_torch | left | 4 | 47% | 49% | 5.99 | 6.17 | 5.81 | 16.49 | 22.2 | 0.79 | 0.38 | 0.01 | -0.02 | 0.56 | 14 | 29.8 | fit plateau (2 levels < 2% gain) |
| patient_10_noreg_hilum_wall | fit_sym | left | 4 | 47% | 44% | 5.94 | 6.64 | 5.25 | 21.11 | 17.5 | 0.71 | 0.33 | -0.14 | 0.00 | 0.55 | 10 | 23.6 | fit plateau (2 levels < 2% gain) |
| patient_10_noreg_hilum_wall | fit_sym_k0.001 | left | 14 | 47% | 46% | 3.97 | 4.35 | 3.60 | 11.17 | 15.2 | 0.77 | 0.69 | -0.17 | -0.07 | 0.29 | 12 | 24.8 | fit plateau (2 levels < 2% gain) |
| patient_10_noreg_hilum_wall | fit_sym_k0.003 | left | 25 | 47% | 46% | 3.39 | 3.69 | 3.09 | 8.52 | 10.1 | 0.87 | 1.25 | -0.39 | -0.21 | 0.32 | 11 | 32.2 | levels exhausted |
| patient_2_noreg_hilum | fit_grad_ap | left | 2 | 86% | 98% | 27.15 | 8.87 | 45.44 | 20.28 | 25.4 | 3.00 | -1.11 | 0.00 | 0.00 | 1.00 | 0 | 0.3 | levels exhausted |
| patient_2_noreg_hilum | fit_grad_ap_sym | left | 2 | 86% | 87% | 12.03 | 11.88 | 12.18 | 26.51 | 35.4 | 1.55 | 0.57 | -0.00 | -0.00 | 1.00 | 0 | 0.3 | levels exhausted |
| patient_2_noreg_hilum | fit_grad_apsi | left | 3 | 86% | 99% | 27.09 | 8.68 | 45.51 | 19.02 | 28.7 | 3.00 | -0.86 | 2.99 | 0.00 | 1.00 | 0 | 0.3 | levels exhausted |
| patient_2_noreg_hilum | fit_grad_apsi_sym | left | 3 | 86% | 85% | 9.80 | 9.40 | 10.21 | 23.35 | 38.4 | 1.49 | 0.51 | -0.28 | -0.00 | 1.00 | 0 | 0.3 | levels exhausted |
| patient_2_noreg_hilum | fit_torch | left | 1 | 86% | 98% | 27.08 | 8.93 | 45.23 | 20.65 | 24.0 | 3.00 | 0.00 | 0.00 | 0.00 | – | 0 | 0.1 | fit plateau (2 levels < 2% gain) |
| patient_2_noreg_hilum | fit_sym | left | 25 | 86% | 84% | 5.00 | 4.92 | 5.07 | 13.22 | 43.9 | 1.87 | 1.06 | -0.65 | 0.06 | 0.14 | 0 | 0.8 | levels exhausted |
| patient_2_noreg_hilum_wall | fit_torch | left | 1 | 86% | 98% | 27.08 | 8.93 | 45.23 | 20.65 | 24.0 | 3.00 | 0.00 | 0.00 | 0.00 | – | 0 | 0.1 | fit plateau (2 levels < 2% gain) |
| patient_3_noreg_hilum | fit_grad_ap | left | 2 | 37% | 48% | 8.51 | 8.84 | 8.18 | 20.22 | 12.1 | 0.82 | 0.15 | 0.00 | -0.00 | 1.00 | 0 | 0.1 | levels exhausted |
| patient_3_noreg_hilum | fit_grad_ap_sym | left | 2 | 37% | 39% | 8.06 | 9.27 | 6.85 | 26.69 | 9.4 | 0.67 | 0.10 | 0.00 | 0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| patient_3_noreg_hilum | fit_grad_apsi | left | 3 | 37% | 47% | 8.54 | 8.88 | 8.20 | 19.71 | 14.2 | 0.80 | 0.16 | -0.03 | -0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| patient_3_noreg_hilum | fit_grad_apsi_sym | left | 3 | 37% | 39% | 8.07 | 9.30 | 6.84 | 26.61 | 9.6 | 0.67 | 0.09 | -0.02 | 0.00 | 1.00 | 0 | 0.2 | levels exhausted |
| patient_3_noreg_hilum | fit_torch | left | 40 | 37% | 39% | 2.88 | 3.14 | 2.62 | 9.69 | 10.0 | 0.61 | 0.40 | -0.37 | 0.18 | 0.10 | 0 | 0.6 | levels exhausted |
| patient_3_noreg_hilum | fit_sym | left | 40 | 37% | 37% | 2.67 | 3.17 | 2.16 | 9.82 | 8.5 | 0.56 | 0.30 | -0.30 | 0.16 | 0.07 | 0 | 0.7 | levels exhausted |
| patient_3_noreg_hilum | fit_sym_k0.001 | left | 40 | 37% | 37% | 2.59 | 3.05 | 2.12 | 10.03 | 8.6 | 0.65 | 0.47 | -0.55 | -0.02 | 0.21 | 0 | 5.7 | levels exhausted |
| patient_3_noreg_hilum | fit_sym_k0.003 | left | 25 | 37% | 37% | 2.54 | 3.04 | 2.04 | 10.55 | 7.3 | 0.70 | 0.57 | -0.86 | -0.08 | 0.27 | 0 | 6.3 | target error reached |
| patient_3_noreg_hilum_wall | fit_grad_ap_sym | left | 2 | 37% | 41% | 7.92 | 8.94 | 6.90 | 25.47 | 12.1 | 0.71 | 0.19 | 0.00 | 0.00 | 1.00 | 2 | 4.5 | levels exhausted |
| patient_3_noreg_hilum_wall | fit_torch | left | 4 | 37% | 39% | 5.67 | 6.02 | 5.32 | 15.62 | 13.1 | 0.62 | 0.25 | -0.43 | 0.08 | 0.64 | 32 | 64.7 | fit plateau (2 levels < 2% gain) |
| patient_3_noreg_hilum_wall | fit_sym | left | 4 | 37% | 36% | 6.25 | 7.12 | 5.37 | 18.12 | 10.4 | 0.60 | 0.12 | -0.27 | 0.03 | 0.71 | 12 | 27.2 | fit plateau (2 levels < 2% gain) |
| patient_3_noreg_hilum_wall | fit_sym_k0.001 | left | 14 | 37% | 38% | 3.61 | 4.14 | 3.08 | 12.16 | 9.1 | 0.64 | 0.33 | -0.55 | -0.02 | 0.33 | 12 | 28.0 | fit plateau (2 levels < 2% gain) |
| patient_3_noreg_hilum_wall | fit_sym_k0.003 | left | 40 | 37% | 37% | 2.75 | 3.30 | 2.20 | 11.00 | 7.3 | 0.71 | 0.65 | -0.94 | -0.05 | 0.34 | 23 | 58.4 | levels exhausted |


## Per variant (median over patients)

| variant | n | ASSD | fit→tgt | tgt→fit | vol err % | rot ° | ap > 0 | si < 0 | rel ap | rel si | mean p/E vs collapse c | min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| noreg / hilum only / no wall / gradient ap / one-sided | 15 | 8.51 | 8.59 | 8.61 | -10.6 | 17.5 | 11/15 | 10/15 | 0.14 | -0.00 | -0.53 + 4.05·c (max res 1.45) | 0.2 |
| noreg / hilum only / no wall / gradient ap / symmetric | 15 | 8.31 | 9.18 | 7.93 | -2.2 | 12.8 | 14/15 | 9/15 | 0.16 | -0.00 | -0.00 + 1.82·c (max res 0.20) | 0.2 |
| noreg / hilum only / no wall / gradient ap+si / one-sided | 15 | 8.06 | 8.21 | 8.15 | -15.6 | 14.2 | 11/15 | 11/15 | 0.16 | -0.10 | -0.52 + 3.97·c (max res 1.41) | 0.2 |
| noreg / hilum only / no wall / gradient ap+si / symmetric | 15 | 7.84 | 8.35 | 7.09 | -1.3 | 12.0 | 13/15 | 13/15 | 0.17 | -0.09 | 0.01 + 1.75·c (max res 0.19) | 0.2 |
| noreg / hilum only / no wall / regions / one-sided | 15 | 2.95 | 3.03 | 3.08 | -3.4 | 10.7 | 14/15 | 10/15 | 0.23 | -0.28 | -0.62 + 3.86·c (max res 0.97) | 0.4 |
| noreg / hilum only / no wall / regions / symmetric | 43 | 2.59 | 2.85 | 2.53 | 0.9 | 8.6 | 42/43 | 36/43 | 0.56 | -0.36 | -0.09 + 2.02·c (max res 0.23) | 5.3 |
| noreg / hilum only / wall / gradient ap / symmetric | 14 | 7.14 | 7.94 | 6.33 | 0.4 | 13.7 | 13/14 | 10/14 | 0.41 | -0.00 | -0.03 + 1.83·c (max res 0.20) | 5.8 |
| noreg / hilum only / wall / regions / one-sided | 15 | 5.67 | 6.02 | 5.32 | -3.2 | 14.3 | 13/15 | 7/15 | 0.38 | 0.00 | -0.63 + 3.80·c (max res 0.99) | 21.3 |
| noreg / hilum only / wall / regions / symmetric | 46 | 4.39 | 4.83 | 3.87 | 4.1 | 8.5 | 46/46 | 36/46 | 0.68 | -0.28 | 0.10 + 1.31·c (max res 0.35) | 26.3 |


## Paired comparisons (B − A, median over patients; negative error = B better)

| comparison | fixed | n | B better (ASSD) | Δ ASSD | Δ fit→tgt | Δ tgt→fit | Δ rot ° | Δ mean p/E | Δ ap | Δ failed | Δ min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| wall: False → True | noreg, hilum only, gradient ap, symmetric | 14 | 13/14 | -0.78 | -0.67 | -0.75 | 0.5 | -0.03 | 0.11 | 2 | 5.6 |
| wall: False → True | noreg, hilum only, regions, one-sided | 15 | 4/15 | 1.41 | 1.70 | 1.52 | 0.6 | -0.02 | -0.01 | 11 | 20.6 |
| wall: False → True | noreg, hilum only, regions, symmetric | 138 | 20/138 | 1.69 | 2.01 | 1.26 | -0.0 | -0.06 | 0.02 | 11 | 19.1 |
| model: regions → gradient ap+si | noreg, hilum only, no wall, one-sided | 45 | 11/45 | 3.69 | 3.61 | 3.13 | 1.4 | 0.05 | -0.08 | 0 | -0.1 |
| model: regions → gradient ap+si | noreg, hilum only, no wall, symmetric | 101 | 13/101 | 4.80 | 5.50 | 4.30 | 1.1 | 0.00 | -0.21 | 0 | -3.2 |
| model: regions → gradient ap+si | noreg, hilum only, wall, symmetric | 46 | 5/46 | 2.71 | 3.34 | 2.56 | 3.2 | 0.02 | -0.17 | -8 | -19.0 |
| model: regions → gradient ap | noreg, hilum only, no wall, one-sided | 45 | 11/45 | 3.69 | 3.61 | 3.13 | 1.4 | 0.05 | -0.08 | 0 | -0.1 |
| model: regions → gradient ap | noreg, hilum only, no wall, symmetric | 101 | 13/101 | 4.80 | 5.50 | 4.30 | 1.1 | 0.00 | -0.21 | 0 | -3.2 |
| model: regions → gradient ap | noreg, hilum only, wall, symmetric | 46 | 5/46 | 2.71 | 3.34 | 2.56 | 3.2 | 0.02 | -0.17 | -8 | -19.0 |
| model: gradient ap → gradient ap+si | noreg, hilum only, no wall, one-sided | 45 | 11/45 | 3.69 | 3.61 | 3.13 | 1.4 | 0.05 | -0.08 | 0 | -0.1 |
| model: gradient ap → gradient ap+si | noreg, hilum only, no wall, symmetric | 101 | 13/101 | 4.80 | 5.50 | 4.30 | 1.1 | 0.00 | -0.21 | 0 | -3.2 |
| model: gradient ap → gradient ap+si | noreg, hilum only, wall, symmetric | 46 | 5/46 | 2.71 | 3.34 | 2.56 | 3.2 | 0.02 | -0.17 | -8 | -19.0 |
| loss: one-sided → symmetric | noreg, hilum only, no wall, gradient ap | 15 | 15/15 | -0.45 | 0.19 | -1.32 | -1.4 | -0.15 | -0.02 | 0 | 0.0 |
| loss: one-sided → symmetric | noreg, hilum only, no wall, gradient ap+si | 15 | 15/15 | -0.47 | 0.35 | -1.36 | -1.3 | -0.13 | -0.01 | 0 | -0.0 |
| loss: one-sided → symmetric | noreg, hilum only, no wall, regions | 43 | 33/43 | -0.22 | 0.04 | -0.50 | -1.6 | -0.03 | 0.13 | 0 | 5.0 |
| loss: one-sided → symmetric | noreg, hilum only, wall, regions | 46 | 37/46 | -1.02 | 0.10 | -1.66 | -5.1 | -0.03 | 0.20 | 0 | 3.8 |


## Pressure pattern vs collapse and clamp (Pearson r over patients, per variant)

| variant | n | rel ap ~ collapse | rel si ~ collapse | rel ap ~ clamp | rel si ~ clamp | collapse ~ clamp |
|---|---|---|---|---|---|---|
| noreg / hilum only / no wall / gradient ap / one-sided | 15 | -0.25 | -0.19 | 0.11 | 0.23 | -0.47 |
| noreg / hilum only / no wall / gradient ap / symmetric | 15 | 0.45 | -0.18 | -0.28 | 0.14 | -0.47 |
| noreg / hilum only / no wall / gradient ap+si / one-sided | 15 | -0.31 | 0.47 | 0.18 | -0.42 | -0.47 |
| noreg / hilum only / no wall / gradient ap+si / symmetric | 15 | 0.38 | -0.01 | -0.19 | -0.02 | -0.47 |
| noreg / hilum only / no wall / regions / one-sided | 15 | -0.22 | 0.28 | 0.22 | -0.13 | -0.47 |
| noreg / hilum only / no wall / regions / symmetric | 43 | 0.03 | 0.22 | 0.08 | 0.29 | -0.38 |
| noreg / hilum only / wall / gradient ap / symmetric | 14 | 0.13 | 0.02 | -0.18 | 0.14 | -0.31 |
| noreg / hilum only / wall / regions / one-sided | 15 | -0.27 | 0.37 | 0.18 | -0.13 | -0.47 |
| noreg / hilum only / wall / regions / symmetric | 46 | 0.19 | 0.19 | -0.06 | 0.17 | -0.32 |
