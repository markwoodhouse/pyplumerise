# Changelog

## 1.0.0 - 2026-10-17

First stable release of pyPlumeRise.

### Highlights

- Integrates dry, wet, and ice plume formulations with configurable physical
  and solver parameters.
- Supports Wyoming radiosonde and ERA5 meteorological profiles.
- Provides a Python API, interactive dashboard, example notebook, and ensemble
  simulation tools.
- Adds a user guide covering configuration, output variables, plotting, and
  ensemble sampling.
- Improves dashboard meteorological-data handling and ensemble input sampling.
- Supports Python 3.10 and newer; the CI matrix tests Python 3.10, 3.12, and
  3.14.

### Notes

- Dashboard initialization and Wyoming radiosonde retrieval require network
  access to the Wyoming Weather Web service.
- Third-party data and assets are not covered by the BSD-3-Clause license;
  see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- `set_IC()` currently initializes liquid-water and ice mass fractions to zero;
  non-zero initial values for those phases are not supported.
