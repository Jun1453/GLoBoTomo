# GLoBoTomo (P,S,ScS)
A global mantle velocity tomographic model obtained from long-period P, S, ScS travel times.

## Agenda
- [x] Parameterization (Jun)
- [ ] Ray tracing (Jun)
	- [x] 1D velocitry model (Jun)
	- [ ] Sensitivity kernel matrix (Jun)
	- [ ] Summary ray QC (Jun)
- [ ] Correction
	- [ ] Surface wave contrained model for upper mantle (Jun)
	- [ ] Crust (Takumi)
	- [ ] Ellipticity (Takumi)
- [ ] Weighting (Takumi)
- [ ] LSQR Inversion (Jun)

## Modelization
- P and S catalog: [Su et al. (2025)](http://doi.org/10.22541/essoar.174802873.35537594/v1)
- ScS catalog: Matsunaga et al. (prep)
- Probability threshold: 0.85 (P, S)
- Grid: ~HMSL (4º equiareal), double HMSL (2º equiareal)
- Initial velocity model: PREM, AK135
- Sensitivity matrices obtained: G4P, G4A, G2P, G2A
	- WIP: G4P - 40000/1057538 (20000/514580 + 20000/542958)
	- Planned: G4A, G2P, G2A
- Surface wave data:
	- WIP: Using [Houser et al. (2008)](http://doi.org/10.1111/j.1365-246X.2008.03763.x)
	- Planned: Using [Moulik & Ekström (2014)](http://doi.org/10.5281/zenodo.8357379)