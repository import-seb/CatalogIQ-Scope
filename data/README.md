# Data

Place the Product Owner's supplied CSVs (extract ZIP archives first) in `provided/`:

- `Selfcare_Training_data (1).csv`
- `Selfcare_Target_data (1).csv`

Keep these filenames unchanged: the existing notebooks and shared path constants
use them. Source data must remain unchanged and must not be committed without
permission to redistribute it.

Use `interim/` for intermediate data and `processed/` for derived datasets when
needed. All three data directories are ignored by Git; create them as needed.
Document transformations in the notebooks and record agreed rules in
[the decision log](../docs/decisions.md).
