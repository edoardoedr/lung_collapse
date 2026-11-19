# Emphysema: CT index against the visual check (2026-11-19)

The index LAA%−950 (`scripts/ct_density.py`, median-filtered; analysis/results.md 5j) against what the user sees on the CT in 3D Slicer (`scripts/slicer_emphysema.py`, map from `scripts/emphysema_map.py`). Contralateral = the inflated lung of the same scan; collapsed = the air still trapped in the operated lung. Sorted by the index.

Visual: none / mild / moderate / severe, and where (upper / lower lobes, diffuse, bullae).

| Patient | LAA%−950 contralateral | LAA%−950 collapsed | low-attenuation clusters ≥ 1 mL / ≥ 5 mL (% of the lung in them) | Index says | Visual (user) | Notes |
|---|---|---|---|---|---|---|
| karl08 | 64.7 | 32.3 | 1 / 1 (64.1 %) | severe (upper lobe 77 %) | | |
| karl17 | 35.8 | 27.1 | 1 / 1 (34.8 %) | emphysema | | |
| patient_3 | 32.8 | 20.6 | 3 / 1 (30.6 %) | emphysema | | |
| karl06 | 22.8 | 17.3 | 2 / 1 (22.7 %) | emphysema | | |
| karl02 | 13.7 | 11.5 | 39 / 11 (12.1 %) | moderate | **yes: emphysema with bullae** (user) | |
| karl20 | 11.3 | 20.9 | 12 / 4 (9.8 %) | moderate (upper lobe 18 %) | **yes: emphysema with bullae** (user) | |
| karl04 | 7.3 | 7.0 | 4 / 2 (4.9 %) | mild (upper lobe 12 %) | **yes: small emphysema** (user) | |
| patient_1 | 4.2 | 0.6 | 9 / 2 (1.6 %) | none / borderline | **no** (user) | |
| karl10 | 4.0 | 2.3 | 8 / 2 (3.1 %) | none / borderline | **yes: large peripheral bullae** (user) | |
| karl24 | 2.4 | 3.1 | 9 / 2 (2.0 %) | none | **bullae** (user, 2026-11-19) | real but small in volume |
| karl01 | 2.0 | 1.9 | 9 / 1 (1.2 %) | none | **no** (user) | |
| karl23 | 1.9 | 1.1 | 2 / 2 (2.2 %) | none | **no** (user) | the dark clusters lie outside the lung: segmentation error of the contralateral lobes |
| patient_10 | 1.0 | 0.3 | 5 / 2 (0.5 %) | none | **some emphysema** (user, 2026-11-19, correcting the first answer) | 11.3 % unfiltered (noise), 1.0 % filtered, 5 clusters ≥ 1 mL |
| karl18 | 0.8 | 0.0 | 2 / 1 (0.7 %) | none | **no** (user: tiny ones only) | |
| karl19 | 0.7 | 0.0 | 4 / 0 (0.5 %) | none | **no** (user: tiny ones only) | |
| patient_2 | 0.0 | 8.1 | 0 / 0 (0.0 %) | none (collapsed lung almost airless) | | |

Threshold used for "emphysematous" (`physics.barrier` trapped air, results 5j): LAA%−950 ≥ 5 %. If the visual check disagrees, adjust the threshold or the filter and rerun `ct_density.py` (then `--write-config`).

**First comparison (user, 2026-11-19).** karl24 has emphysema bullae and patient_10 some emphysema, both below the 5 % threshold (2.4 % and 1.0 %). The index measures the *volume* fraction: a few bullae are a few tens of mL, 1–3 % of the lung, while they are evident to the eye. The border is not the cause (with 1 voxel instead of 2 mm of erosion: karl24 2.5 %). Counting compact clusters ≥ 1 mL (bullae) does not separate them either: karl23, karl18, karl19 (index: none) have 2–4 clusters ≥ 1 mL too. Needed: the visual verdict on those, to know whether the clusters are bullae or something else (residual airways, vessels' gaps).

For the model the volume is what counts: φ = φ_tissue + r f_e is continuous in f_e, and 2 % of bullae changes φ by ~0.01. The 5 % threshold only selects the patients on which r is estimated (they need measurable emphysema). Bullae probably trap air more than diffuse emphysema (r close to 1 inside them): an open point.

**Second comparison (user, 2026-11-19): the index cannot separate the mild cases.** Visual yes / no against the index (contralateral LAA%−950, clusters ≥ 1 mL):
- yes: patient_10 (1.0 %, 5 clusters), karl24 (2.4 %, 9), karl10 (4.0 %, 8; large peripheral bullae);
- no: karl19 (0.7 %, 4), karl18 (0.8 %, 2), karl23 (1.9 %, 2: outside the lung, a segmentation error), karl01 (2.0 %, 9), patient_1 (4.2 %, 9).

The two groups overlap on both measures (patient_1 4.2 % no / karl10 4.0 % yes; karl01 2.0 % no / patient_10 1.0 % yes), so no threshold on the index reproduces the visual verdict below ~5 %. Above it (karl08, karl17, patient_3, karl06, karl02, karl20, karl04: 7–65 %) the index is unambiguous (visual check of these pending). **Decision proposed:** the model uses the volume fraction (continuous, φ changes by < 0.02 below 4 %); the clinical label emphysema yes / no comes from the visual check (this table).

**Decision (user, 2026-11-19):** emphysema yes / no is an input given by the user (`physics.emphysema`), not a threshold on the index; the barrier is switched on by the user too (`physics.barrier`). Set in the configs: **yes** karl02, karl04, karl06, karl08, karl10, karl17, karl20, karl24, patient_3, patient_10; **no** karl01, karl18, karl19, karl23, patient_1, patient_2. karl06, karl08, karl17, patient_3 (index 23–65 %) and patient_2 (0 %, collapsed lung airless) were set from the index and confirmed by the user (2026-11-19).
