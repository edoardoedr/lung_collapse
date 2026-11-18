"""Predict the collapsed lung of one patient: the inflated lung (with airways / vessels for the hilum)
is collapsed by a forward run with all the physics given. No collapsed lung is needed; with one in
the config, the errors against it are reported. The code is eurelaps_lung/predict.py (also
`python -m eurelaps_lung.predict`, the entry point of the 3D Slicer extension).

    python main_predict.py --config configs/<p>.json --collapse 40 --params calibration/E.json [--frames 30]
    python main_predict.py --config configs/<p>.json --collapse 40 --from-fit fitE
    python main_predict.py --config configs/<p>.json --collapse 40          # the config's physics
    python main_predict.py --config configs/<p>.json --collapse target      # the collapse % of inputs.collapsed

Parameters (later ones win):
  --params FILE    rule file of main_calibrate.py: E (and tissue_density, nu) from the collapse %
                   (and the CT density physics.ct_density for the rules E_rho / sg_rho)
  --from-fit RUN   E, nu, tissue_density of this patient's fit run output_dir/runs/RUN (and its q0
                   when --collapse is not given: the fit reproduced)
  --collapse P     collapse %: q0 is searched so that V_collapsed / V_inflated = 1 - P / 100;
                   'target': P of the collapsed lung of the config; without it, the config's physics.collapse
  --set KEY=VALUE  any config key, e.g. physics.gravity='{"reference":"supine","collapse":"left_lateral"}'

The model geometry (geometry -> hilum -> fem_setup) is built first when output_dir/fem/setup/problem.npz
is missing, or when --steps asks for it. Output: output_dir/runs/<run>/ (surface.vtp, state.npz,
summary.json) and, with --frames > 0, the collapse sequence for 3D Slicer.
"""

from eurelaps_lung.predict import collapse_overrides, fit_run_overrides, lung_extent, main  # noqa: F401

if __name__ == "__main__":
    main()
