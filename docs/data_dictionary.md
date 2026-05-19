# Volve Field — Data Dictionary

Reference for all columns in the Volve daily production dataset, with **oil & gas domain
context** for each. This is the kind of glossary you'd be expected to know in a Halliburton
interview.

## Identification columns

| Column | Description |
|--------|-------------|
| `DATEPRD` | Production date (one row per well per day) |
| `WELL_BORE_CODE` | NPD-prefixed well identifier (e.g., `NO 15/9-F-1 C`) |
| `NPD_WELL_BORE_CODE` | Numeric well code in NPD registry (e.g., `7405`) |
| `NPD_WELL_BORE_NAME` | Standard well name (e.g., `15/9-F-1 C`) |
| `NPD_FIELD_CODE` | Field code in NPD registry |
| `NPD_FIELD_NAME` | `VOLVE` (constant for this dataset) |
| `NPD_FACILITY_CODE` | Facility code (`369304` = MÆRSK INSPIRER) |
| `NPD_FACILITY_NAME` | `MÆRSK INSPIRER` (the production rig) |

### NPD well naming convention

`15/9-F-1 C` means:
- **15/9** — block on the Norwegian continental shelf (Block 15/9)
- **F** — slot or template letter
- **1** — original sondeo (drilling)
- **C** — sidetrack identifier (A = first sidetrack, B = second, C = third)

A "sidetrack" is when the wellbore is deviated to access a different part of the reservoir.
Each sidetrack adds a letter. Wells with multiple sidetracks indicate **geological challenges
or attempts to recover from drilling problems**.

## Operational state

| Column | Description | Units | Notes |
|--------|-------------|-------|-------|
| `ON_STREAM_HRS` | Hours the well was producing/injecting | hours (0-24) | Max 24/day. Lower = downtime |
| `FLOW_KIND` | `production` or `injection` | enum | |
| `WELL_TYPE` | `OP` (oil producer) or `WI` (water injector) | enum | A well can be converted between these |

### Critical concept: Producers vs Injectors

**Oil Producer (`OP`)**: pumps reservoir fluids (oil + gas + water) to surface.
This is what makes money.

**Water Injector (`WI`)**: pumps water INTO the reservoir to:
1. **Maintain pressure** as oil is extracted (pressure naturally drops as reservoir is depleted)
2. **Sweep oil toward producers** — water displaces oil through rock pores

A typical field has a **3:1 or 4:1 producer:injector ratio**. Volve had 5 producers + 2 injectors.

## Pressure & temperature measurements

All pressures are in **bar** (1 bar ≈ 14.5 psi). Temperatures in **°C**.

| Column | Description | What it tells you |
|--------|-------------|-------------------|
| `AVG_DOWNHOLE_PRESSURE` | Bottomhole pressure | Reservoir energy. Higher = more drive |
| `AVG_DOWNHOLE_TEMPERATURE` | Bottomhole temperature | Depth proxy (geothermal gradient) |
| `AVG_DP_TUBING` | Differential pressure tubing | Restriction inside the tubing |
| `AVG_ANNULUS_PRESS` | Annulus pressure | Pressure between tubing and casing |
| `AVG_WHP_P` | Wellhead pressure | Pressure at surface |
| `AVG_WHT_P` | Wellhead temperature | Surface temperature (depends on flow rate) |
| `DP_CHOKE_SIZE` | Pressure differential across choke | Throttling intensity |

### Pressure gradient physics

In a typical oil reservoir:
- **Hydrostatic gradient**: ~0.43 psi/ft (water column)
- **Lithostatic gradient**: ~1.0 psi/ft (rock column)
- Reservoirs are **normally pressured** if their pressure matches hydrostatic
- **Overpressured** reservoirs (above hydrostatic) are dangerous to drill but high-energy

Volve was a **normally pressured** Jurassic sandstone reservoir at ~2,800 m depth.

## Choke (production rate control)

| Column | Description | Units |
|--------|-------------|-------|
| `AVG_CHOKE_SIZE_P` | Choke opening percentage | % (0-100) |
| `AVG_CHOKE_UOM` | Unit of measure for choke | always `%` in Volve |

### What is a choke?

A **choke valve** is a precision orifice at the wellhead that controls flow rate. Operators
adjust the choke to:
- **Increase production**: open the choke wider
- **Conserve reservoir energy**: close the choke partially
- **Prevent sand production**: keep velocity below critical threshold
- **Manage water cut**: throttle high-water-cut wells

A fully open choke (100%) is **not always optimal** — it can damage the reservoir long-term.

## Production volumes (the targets)

All in **m³** (cubic meters). For barrel conversion: **1 m³ = 6.29 barrels**.

| Column | Description |
|--------|-------------|
| `BORE_OIL_VOL` | **Oil produced that day** ← model target |
| `BORE_GAS_VOL` | Gas produced (Sm³ — standard cubic meters at surface conditions) |
| `BORE_WAT_VOL` | Water produced |
| `BORE_WI_VOL` | Water injected (only populated for injector wells) |

### Key engineered ratios

**Water Cut** = `BORE_WAT_VOL / (BORE_OIL_VOL + BORE_WAT_VOL)`

- 0% = pure oil production (new well, ideal)
- 50% = half the liquid is water (typical late-life)
- 90%+ = "watered out" — economic limit approaching

**Gas-Oil Ratio (GOR)** = `BORE_GAS_VOL / BORE_OIL_VOL`

Volve was an oil reservoir with relatively low GOR (~100-500 scf/bbl). High GOR wells produce
mostly gas with little oil (different play type — "gas-condensate").

**Oil per hour** = `BORE_OIL_VOL / ON_STREAM_HRS`

Normalized production rate. Two wells producing the same `BORE_OIL_VOL` are very different if
one did it in 4 hours vs the other in 24 hours.

## Decline curves (the physics behind production prediction)

Oil wells produce a lot at first, then decline. The mathematical model is the **Arps decline
curve** (J.J. Arps, 1945):

```
q(t) = q_i * (1 + b * D_i * t) ^ (-1/b)
```

Where:
- `q(t)` = production rate at time `t`
- `q_i` = initial production rate
- `D_i` = initial decline rate
- `b` = decline exponent (0 = exponential, 1 = harmonic, between = hyperbolic)

This formula is why **`well_age_days` is such a predictive feature** in ML models.

## References

- **NPD (Norwegian Petroleum Directorate)**: https://www.sodir.no/en/
- **Volve press release** (Equinor, 2018): describes the release decision
- **SPE petrowiki on Arps decline**: https://petrowiki.spe.org/Production_forecasting_decline_curve_analysis
