# TransFit Multi-band Photometry Redesign

## Status

- Document type: design proposal
- Scope: multi-band photometry only
- Covers:
  - filter definitions and registry
  - Vega / AB magnitude systems
  - explicit distance inputs for nearby objects
  - extinction handling
  - internal photometry pipeline
  - serialization and backward compatibility
- Does not change bolometric fitting design


## 1. Background and Problems

The current multi-band implementation in [transfit/api.py](C:/Users/zyh/Desktop/TransFit/transfit/api.py), [transfit/modules/sed/blackbody.py](C:/Users/zyh/Desktop/TransFit/transfit/modules/sed/blackbody.py), [transfit/modules/io.py](C:/Users/zyh/Desktop/TransFit/transfit/modules/io.py), and [transfit/modules/plot.py](C:/Users/zyh/Desktop/TransFit/transfit/modules/plot.py) works for a simple AB-magnitude workflow, but it has four structural limitations:

1. `filters` only means `band -> nu_eff`, which is not rich enough for Vega support or future bandpass integration.
2. Distance is effectively tied to `z` at the public API level, which is not acceptable for nearby sources with independent distance measurements.
3. Extinction is not modeled as a first-class part of the photometry pipeline.
4. The internal meaning of "mag" is overloaded: the code computes flux-like quantities internally but exposes only AB magnitude as the default magnitude system.

The redesign should fix these issues without making the public API difficult to learn.


## 2. Design Goals

### 2.1 User-facing goals

- Keep the public API simple.
- Preserve the current shorthand filter style for easy AB usage.
- Add Vega magnitudes without forcing users to learn internal classes.
- Allow explicit luminosity distance inputs alongside, or instead of, `z`.
- Allow users to provide extinction in a straightforward way.

### 2.2 Developer-facing goals

- Centralize filter logic in one subsystem.
- Make extinction a separate subsystem, not an ad hoc option in multiple files.
- Use a single internal photometry pipeline for forward models, fitting, plotting, and saved results.
- Keep room for future bandpass integration without redesigning the public API again.

### 2.3 Non-goals for the first implementation phase

- Full synthetic photometry over arbitrary bandpasses is not required in phase 1.
- A large built-in filter catalog is not required in phase 1.
- Bolometric APIs do not need new public options.


## 3. High-level Architecture

The multi-band photometry stack will be split into three layers:

1. `filters`
   - Defines what a filter is
   - Resolves user input into normalized filter objects
   - Owns Vega zero points and built-in presets

2. `extinction`
   - Defines extinction specifications
   - Resolves user input into normalized extinction objects
   - Applies extinction in the correct frame

3. `photometry`
   - Converts model outputs `Teff`, `Rph`, `Lbol` into observer-frame fluxes
   - Applies distance, redshift, extinction, and magnitude conversions

This leads to the following data flow:

1. Physical model returns `t_s`, `Lbol`, `Teff`, `Rph`
2. Photometry layer builds rest-frame spectral quantities
3. Distance and redshift map them to observer-frame flux density
4. Extinction modifies the observer/model flux
5. Output is converted to:
   - `flux`
   - `AB magnitude`
   - `Vega magnitude`


## 4. Package Layout

The new module layout should be:

```text
transfit/
  modules/
    filters/
      __init__.py
      core.py
      registry.py
      normalize.py
      serde.py
    extinction/
      __init__.py
      core.py
      normalize.py
      serde.py
    photometry.py
```

### 4.1 Responsibilities

- `modules/filters/core.py`
  - internal filter data structures
- `modules/filters/registry.py`
  - built-in preset definitions
- `modules/filters/normalize.py`
  - parse and validate public `filters=...`
- `modules/filters/serde.py`
  - convert normalized filters to and from saved-result payloads
- `modules/extinction/core.py`
  - internal extinction data structures
- `modules/extinction/normalize.py`
  - parse and validate public `extinction=...`
- `modules/extinction/serde.py`
  - convert normalized extinction specs to and from saved-result payloads
- `modules/photometry.py`
  - flux-space evaluation and conversions


## 5. Public API Design

The public multi-band APIs should stay shallow.

### 5.1 Target signatures

```python
fit_multiband(
    *,
    data,
    model,
    z=None,
    distance_modulus=None,
    filters,
    y_kind="mag",
    mag_system="ab",
    extinction=None,
    priors=None,
    fixed=None,
    sampler="emcee",
    sampler_kwargs=None,
    model_kwargs=None,
)
```

```python
lightcurve_multiband(
    *,
    model,
    params=None,
    z=None,
    distance_modulus=None,
    filters,
    bands,
    y_kind="mag",
    mag_system="ab",
    extinction=None,
    t_max_days=150.0,
    sed=None,
    solver_kwargs=None,
)
```

```python
predict_multiband(
    *,
    model,
    params=None,
    z=None,
    distance_modulus=None,
    filters,
    t_days,
    band,
    y_kind="mag",
    mag_system="ab",
    extinction=None,
    t_max_days=150.0,
    interp_fill="nan",
    sed=None,
    solver_kwargs=None,
)
```

### 5.2 Public argument meanings

- `y_kind`
  - `"mag"` or `"flux"`
  - controls the observation space returned to the user and used in the likelihood
- `mag_system`
  - `"ab"` or `"vega"`
  - only meaningful when `y_kind="mag"`
- `z`
  - used for time dilation and frequency redshift
  - not required to define distance if the user supplies an explicit distance
- `distance_modulus`
  - explicit distance option
  - used for flux normalization when supplied
- `filters`
  - user filter definitions or built-in preset references
- `extinction`
  - optional extinction specification


## 6. Filter Subsystem Design

### 6.1 Internal normalized object

All filter input forms must be normalized into a single immutable internal type.

```python
@dataclass(frozen=True)
class FilterProfile:
    label: str
    filter_id: str
    kind: Literal["mono", "bandpass"]
    source: Literal["builtin", "user", "legacy"]
    detector: Literal["energy", "photon"] = "energy"

    nu_eff_hz: float | None = None
    wavelength_A: np.ndarray | None = None
    throughput: np.ndarray | None = None

    zero_points_jy: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
```

### 6.2 Meaning of fields

- `label`
  - the band key used by the user data, such as `"B"` or `"ztfg"`
- `filter_id`
  - physical filter identity, such as `"johnson_cousins.B"` or `"legacy:B"`
- `kind`
  - `"mono"` for effective-frequency approximation
  - `"bandpass"` for full throughput curves
- `source`
  - `"builtin"` for registry filters
  - `"user"` for explicit user definitions
  - `"legacy"` for old `band -> float` shorthand
- `zero_points_jy`
  - may contain `"vega"` and possibly other systems later
  - AB does not need to be stored per filter because it is globally 3631 Jy

### 6.3 Supported public filter input forms

The public API should support exactly three styles.

#### A. Legacy shorthand

```python
filters = {
    "g": 6.3e14,
    "r": 4.8e14,
}
```

Meaning:
- a legacy mono-frequency filter
- valid for AB workflows
- not enough by itself for Vega unless the user switches to an explicit filter form

#### B. Built-in preset reference

```python
filters = {
    "B": "johnson_cousins.B",
    "V": "johnson_cousins.V",
}
```

Meaning:
- the filter definition is loaded from the built-in registry
- this is the recommended style for Vega use

#### C. Explicit custom filter definition

Mono-frequency custom filter:

```python
filters = {
    "B": {
        "nu_eff_hz": 6.8e14,
        "vega_zero_point_jy": 4260.0,
    }
}
```

Future full bandpass form:

```python
filters = {
    "myband": {
        "wavelength_A": wave,
        "throughput": resp,
        "vega_zero_point_jy": 3500.0,
    }
}
```

### 6.4 Public rules

- Band labels remain case-sensitive.
- The user is not required to learn `FilterProfile`.
- The user does not pass `kind`, `source`, or `detector` in common workflows.
- Vega support is attached to filters, not to a top-level `vega_zero_points_jy` argument.

### 6.5 Built-in filter registry

Built-in filters must be identified by stable IDs, for example:

- `johnson_cousins.B`
- `johnson_cousins.V`
- `sdss.g`
- `sdss.r`
- `ztf.g`
- `ztf.r`

Each built-in entry should store:

- `filter_id`
- `kind`
- `nu_eff_hz` or bandpass arrays
- `zero_points_jy`
- `meta.reference`
- `meta.version`

### 6.6 What the software must never do

The software must not:

- infer Vega zero points from `nu_eff_hz`
- infer Vega zero points from band labels like `"B"` or `"V"` alone
- silently guess a filter system from a short label


## 7. Distance Design

### 7.1 Public rule

The user may supply:

- `z` only
- explicit distance only
- both `z` and explicit distance

### 7.2 Internal behavior

- `z` controls observer/rest-frame mapping
- explicit distance controls flux normalization
- if `z` is omitted, use `z = 0.0`
- if explicit distance is omitted, derive it from `z`
- if both are supplied, use:
  - `z` for redshift effects
  - explicit distance for flux normalization

### 7.3 Validation

The public explicit distance input is:

- `distance_modulus`

Internally normalize them into:

```python
@dataclass(frozen=True)
class DistanceSpec:
    z: float
    DL_cm: float
    source: Literal["from_z", "distance_modulus"]
```

Older saved results may still contain `"DL_cm"` as a legacy source tag, but that should no longer appear in the public API.


## 8. Extinction Design

### 8.1 Public API

Keep a single top-level argument:

```python
extinction=None
```

### 8.2 Supported public forms

#### A. Per-band extinction map

```python
extinction = {
    "B": 0.42,
    "V": 0.31,
}
```

Meaning:
- each value is `A_band` in magnitudes
- interpreted in the observer frame
- this is the simplest and most important user-upload case

#### B. Structured dust-law input

```python
extinction = {
    "mw": {"E_BV": 0.03, "R_V": 3.1, "law": "fitzpatrick99"},
    "host": {"E_BV": 0.10, "R_V": 3.1, "law": "fitzpatrick99"},
}
```

Meaning:
- `mw` acts in the observer frame
- `host` acts in the rest frame

### 8.3 Internal normalized objects

```python
@dataclass(frozen=True)
class BandExtinction:
    values_mag: dict[str, float]
    frame: Literal["observer"] = "observer"
```

```python
@dataclass(frozen=True)
class DustLawComponent:
    name: str
    frame: Literal["observer", "rest"]
    law: str
    E_BV: float
    R_V: float
```

```python
@dataclass(frozen=True)
class ExtinctionSpec:
    band_map: BandExtinction | None = None
    components: tuple[DustLawComponent, ...] = ()
```

### 8.4 Rules

- Per-band `A_band` extinction is applied after flux is in the observer frame.
- `mw` law extinction is evaluated using the observer-frame filter definition.
- `host` law extinction is evaluated using rest-frame wavelengths.
- Missing used bands in a per-band extinction map should raise an error.
- Extra unused bands in a per-band extinction map may be ignored.


## 9. Internal Photometry Pipeline

### 9.1 Canonical internal quantity

The canonical internal photometric quantity must be flux density, not magnitude.

The pipeline should compute:

1. rest-frame `L_nu`
2. observer-frame `F_nu`
3. extincted `F_nu`
4. convert to:
   - `flux`
   - `AB mag`
   - `Vega mag`

### 9.2 Why flux is the canonical internal space

- distance acts naturally in flux space
- extinction acts naturally in flux space
- Vega and AB are only different output zero-point conventions
- future bandpass integration is naturally defined in flux space

### 9.3 Likelihood rule

Likelihood should be evaluated in the same observation space as the input data:

- if `y_kind="flux"`, compare model and data in flux space
- if `y_kind="mag"`, compare model and data in magnitude space

This keeps user expectations simple while preserving a clean internal physical pipeline.

### 9.4 Conversion formulas

AB magnitude:

```text
m_AB = -2.5 log10(F_nu / 3631 Jy)
```

Vega magnitude:

```text
m_Vega = -2.5 log10(F_nu / F_nu0_vega)
```

Flux extinction:

```text
F_nu,ext = F_nu * 10^(-0.4 * A_lambda)
```


## 10. Validation Policy

Validation must be strict and explicit.

### 10.1 Filter validation

- all used bands must exist in the normalized filter map
- mono filters require a positive finite `nu_eff_hz`
- bandpass filters require:
  - 1D finite wavelength array
  - 1D finite throughput array
  - same length
  - strictly increasing wavelength
  - non-negative throughput
  - at least one positive throughput value

### 10.2 Magnitude-system validation

- `mag_system` only matters when `y_kind="mag"`
- `mag_system="ab"` always valid
- `mag_system="vega"` requires a Vega zero point for every used band
- a legacy float filter with `mag_system="vega"` must raise an error

### 10.3 Distance validation

- no more than one explicit distance input may be supplied
- all distances must be positive and finite
- if no `z` and no explicit distance are supplied, raise an error

### 10.4 Extinction validation

- per-band extinction values must be finite and non-negative
- `E_BV` must be finite and non-negative
- `R_V` must be finite and positive
- unknown dust laws must raise an error


## 11. Serialization and Saved Results

The saved-result payload should move from "raw user input" to "resolved, reproducible state".

### 11.1 Target saved context shape

```python
{
    "schema_version": 2,
    "distance": {
        "z": ...,
        "DL_cm": ...,
        "source": ...,
    },
    "photometry": {
        "y_kind": "mag",
        "mag_system": "vega",
    },
    "filters": {
        "B": { ... serialized FilterProfile ... },
        "V": { ... serialized FilterProfile ... },
    },
    "extinction": { ... serialized ExtinctionSpec ... },
}
```

### 11.2 Backward compatibility

Old saved results should continue to load.

Backward-compatibility policy:

- if old `ctx["filters"]` is `band -> float`, load it as `source="legacy"` mono filters
- if old `ctx` has only `y_kind`, default `mag_system="ab"`
- if old `ctx["distance"]` has only `z`, derive distance using current cosmology


## 12. Recommended Public Usage

### 12.1 Simple AB workflow

```python
filters = {
    "g": 6.3e14,
    "r": 4.8e14,
}

res = tf.fit_multiband(
    data=data,
    model="nickel",
    z=0.01,
    filters=filters,
    y_kind="mag",
    mag_system="ab",
)
```

### 12.2 Vega workflow with built-in filters

```python
filters = {
    "B": "johnson_cousins.B",
    "V": "johnson_cousins.V",
}

res = tf.fit_multiband(
    data=data,
    model="nickel",
    z=0.002,
    distance_modulus=27.78,
    filters=filters,
    y_kind="mag",
    mag_system="vega",
)
```

### 12.3 User-uploaded extinction coefficients

```python
extinction = {
    "B": 0.42,
    "V": 0.31,
}

res = tf.fit_multiband(
    data=data,
    model="nickel",
    z=0.002,
    distance_modulus=27.78,
    filters=filters,
    y_kind="mag",
    mag_system="vega",
    extinction=extinction,
)
```


## 13. Internal Integration Plan

### Phase 1

- add `modules/filters/` package
- add `modules/extinction/` package
- add `modules/photometry.py`
- keep mono-frequency evaluation only
- support:
  - explicit distance
  - Vega via built-in filters or explicit custom zero points
  - per-band extinction map

### Phase 2

- add parameterized dust-law evaluation
- add built-in preset expansion
- migrate plotting and saving to the new normalized context format

### Phase 3

- add optional full bandpass integration
- keep the same public API


## 14. Testing Plan

Required tests:

- filter normalization
  - legacy float filter
  - built-in preset filter
  - explicit custom mono filter
- Vega validation
  - preset with Vega zero point passes
  - legacy float filter with Vega fails
- distance normalization
  - `z` only
  - explicit distance only
  - `z + explicit distance`
  - conflicting explicit distance inputs fail
- extinction
  - per-band map application
  - missing used band fails
  - structured MW and host components normalize correctly
- photometry
  - AB and Vega outputs differ only by zero-point offset for the same flux
  - `flux` path is independent of `mag_system`
- I/O
  - save/load roundtrip with normalized filters and extinction
  - load legacy result payload
- plotting
  - plot uses stored normalized filters and photometry config


## 15. Final Design Decisions

This redesign adopts the following decisions:

1. Filters become a dedicated subpackage under `modules`.
2. Extinction becomes a dedicated subpackage under `modules`.
3. Internal photometry is always computed in flux space.
4. Magnitude conversion is the final output step only.
5. Vega zero points belong to filters, not to a top-level API argument.
6. Distance is decoupled from redshift in the public API.
7. The public API remains simple:
   - one `filters=` argument
   - one `extinction=` argument
   - one `mag_system=` argument
   - optional explicit distance arguments
8. The software remains strict:
   - no silent filter guessing
   - no silent Vega guessing
   - no silent partial extinction maps
