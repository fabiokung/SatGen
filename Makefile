.PHONY: test lint check smoke tidal-track

test:
	python -m pytest test_evolve_unit.py test_stripping.py test_subhalo_functions.py -v

lint:
	pyright stripping_common.py subhalo_functions.py tidal_track_helpers.py \
	        test_evolve_unit.py test_stripping.py test_subhalo_functions.py

check: lint test

smoke:
	python SatEvo.py --datadir test_data/ --outdir test_data/sat_out/
	python scripts/check_output.py test_data/sat_out/tree*_lgM12.*.npz

tidal-track:
	python test_evolve.py
