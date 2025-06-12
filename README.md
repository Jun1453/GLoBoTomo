# GLoBoTomo (P,S,ScS)
A global mantle velocity tomographic model obtained from long-period P, S, ScS travel times.

## Agenda
- [x] Parameterization (Jun)
- [X] Ray tracing (Jun)
	- [x] 1D velocitry model (Jun)
	- [X] Summary ray QC (Jun)
	- [X] Sensitivity kernel matrix (Jun)
- [ ] Correction
	- [X] Surface wave contrained model for upper mantle (Jun)
	- [ ] Crust (Takumi)
	- [ ] Ellipticity (Takumi)
- [ ] Weighting (Takumi)
- [x] LSQR Inversion (Jun)

## Modelization
- P and S catalog: [Su et al. (2025)](http://doi.org/10.22541/essoar.174802873.35537594/v1)
- ScS catalog: Matsunaga et al. (prep)
- Probability threshold: 0.7 (P) and 0.5 (S)
- Grid: ~HMSL (4º equiareal), double HMSL (2º equiareal)
- Initial velocity model: PREM, AK135
- Sensitivity matrices obtained: G4P, G4A, G2P, G2A
	- WIP: G4P - 917,879 rays for P and 1,191,384 ray for S
	- Planned: G4A, G2P, G2A
- Surface wave data:
	- WIP: Using [Houser et al. (2008)](http://doi.org/10.1111/j.1365-246X.2008.03763.x)
	- Planned: Using [Moulik & Ekström (2014)](http://doi.org/10.5281/zenodo.8357379)