# r16_fixed.py — no-filter run

The filter-masking experiment compares all three controllers with and without the
posthoc HOCBF safety filter. Two build changes are needed vs. the bundled `code/`:
  1. make the DPG filter optional (respect cfg.safety_filter only), in
     dsceos_validation.py: replace
       `if (cfg.safety_filter or cfg.controller == "projected_gradient_hocbf") else None`
     by `if cfg.safety_filter else None` (two occurrences),
  2. widen the numerical containment clip from +-0.05 to +-0.5 (two occurrences of
     `arrays["lower"] - 0.05 ... arrays["upper"] + 0.05`).
Apply these two edits to a COPY of code/ named `code_nofilt/` and point r16_fixed.py's
bootstrap at it. The bundled r16_fixed.json holds the resulting values
(filtered PD J_T 1.584, unfiltered PD 1.895 with an 8.7% capacity excursion;
D-SCEOS and DPG feasible without the filter).
