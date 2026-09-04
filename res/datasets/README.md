# Reproducible datasets

Dataset files are intentionally not versioned. Source metadata and preprocessing
rules live in `configs/datasets/`; the import commands reconstruct the expected
local layout:

```text
res/datasets/
  operations_forecasting_2026/{raw,processed}/
  electricity_load_diagrams/{raw,processed}/
```

Populate either dataset independently from the repository root:

```bash
uv run python -m src.datasets.import_dataset operations
uv run python -m src.datasets.import_dataset electricity
```

| Alias | Canonical directory | Stored table | Feature sets |
| --- | --- | --- | --- |
| `operations` | `operations_forecasting_2026` | CSV | `provided` |
| `electricity` | `electricity_load_diagrams` | Parquet | `raw`, `operations_calendar`, `calendar_extended` |

The manifests define aliases, pinned source metadata, storage formats, validation
horizons, and allowed feature sets. Local chronological splits are created
automatically under `processed/splits/horizon_<N>/` when training or baseline
evaluation starts.

## Representations

The operations importer downloads the provided Hugging Face snapshot at the
revision pinned in its manifest and preserves the supplied CSV/JSON files.

The electricity importer reproduces the Hugging Face `lstnet` configuration from
the original UCI archive: the 15-minute measurements are summed to hourly values,
the interval is restricted to 2012–2014, and clients inactive at the beginning of
2012 are removed. These hourly-resampled target values are preserved without unit
conversion or rescaling. Parquet changes only the storage layout to one
`(series_id, timestamp, target)` row per observation.

The electricity feature columns are generated at load time rather than written
into the base target table. Every engineered value depends only on its timestamp,
so future targets never enter the covariates.

Use `--force` only when the processed representation should be rebuilt:

```bash
uv run python -m src.datasets.import_dataset electricity --force
```
