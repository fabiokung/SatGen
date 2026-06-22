# Galacticus comparison data

`galacticus_subhalo_evolution.ipynb` expects a Galacticus output file at
`analysis/galacticus_subhalos.hdf5` (gitignored; drop your own copy here).

One Galacticus merger tree evolved with full satellite physics, dumped at many
snapshots spanning z = 10 -> 0 (`Outputs/OutputN`, output number rises as z
falls; per-output cosmic time is the `outputTime` attribute on each group).

- **Single host**, the one node isolated (`nodeIsIsolated==1`) to z=0:
  M_vir = 1e8 Msun, r_s ~ 0.33 kpc, c ~ 37 at z=0.
- **One level deep.** Every satellite (`nodeIsIsolated==0`) has the host as its
  parent; no sub-subhalos. Subhalos are tracked across snapshots by `nodeIndex`.
- **Frozen NFW profile.** r_s is set at infall and only `satelliteBoundMass`
  evolves (no tidal heating). So there is no v_max/r_max track in the output --
  the notebook derives a reference one.

## Fields read (per node, in `Outputs/OutputN/nodeData/`)

- `nodeIndex`, `nodeIsIsolated` -- node id; 1 host/central, 0 satellite.
- `basicMass` -- virial mass, frozen at the infall value for satellites.
- `satelliteBoundMass` -- current bound mass (the stripping signal).
- `darkMatterProfileScale` -- NFW scale radius r_s [Mpc].
- `darkMatterOnlyRadiusVirial` -- virial radius r_vir [Mpc]; c = r_vir/r_s.
- `positionOrbital{X,Y,Z}` [Mpc], `velocityOrbital{X,Y,Z}` [km/s] -- orbit rel. host.
- `basicTimeLastIsolated` -- infall time [Gyr].

Units in the file: Mpc, km/s, Msun. The engine converts lengths to kpc and
velocities to kpc/Gyr.
