# Data

The raw file is the AI4I 2020 Predictive Maintenance Dataset from the UCI Machine
Learning Repository (dataset 601, DOI `10.24432/C5HS5C`). It is licensed under
CC BY 4.0.

Expected CSV SHA-256:
`dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e`.

Run `prepare-data` to download the official archive, validate its schema, and
create the initial analysis under `artifacts/eda/`. The downloaded CSV is kept
out of Git and can be recreated from the source.

Only the six operating features listed in `machine_risk.data.FEATURE_COLUMNS`
are model inputs. Identifiers and the published failure targets are excluded.
