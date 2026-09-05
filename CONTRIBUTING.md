# Contributing

Thanks for considering a contribution! This project aims to be a clean, reusable
reference implementation of SAR-based wildfire detection, so readability and test
coverage matter as much as features.

## Development setup

```bash
py -3.13 -m venv venv                     # Windows;  python3.13 -m venv venv elsewhere
venv\Scripts\Activate.ps1                 # or: source venv/bin/activate
python -m pip install --upgrade pip
pip install "torch>=2.2" "torchvision>=0.17" --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
pytest -q                                 # should be green before you start
```

Everything runs on the built-in **synthetic data generator** — no credentials,
no GPU, no downloads required to develop or test.

## Before opening a pull request

Run the same three checks CI runs:

```bash
ruff check .
mypy --ignore-missing-imports src/ app/
pytest -q --cov=src --cov=app
```

- Coverage must stay **≥ 70%** (CI enforces `--cov-fail-under=70`).
- New behaviour needs a test. Put fixtures in `tests/conftest.py` if shared.
- Keep tests offline and CPU-only.

## Code style

- **Formatting & linting**: `ruff` (config in `pyproject.toml`). Line length 100.
- **Types**: full type hints; `mypy` must pass. Prefer `X | None` over `Optional[X]`.
- **Docstrings**: Google style. Every module gets a module docstring explaining
  *what and why* (and the physics/method where relevant); every public
  function/class documents Args / Returns / Raises, plus array **shapes and
  units**. Non-obvious remote-sensing maths gets an inline comment citing the
  formula.
- **Logging, not printing**: use `src.utils.logging.get_logger(__name__)`.
  `print` is only acceptable in `scripts/` banners.
- **Config over constants**: new knobs go in `configs/config.yaml` +
  `src/config.py`, not hard-coded.

## Commit & branch conventions

- Branch off `main`: `feat/…`, `fix/…`, `docs/…`, `refactor/…`, `test/…`.
- Imperative, present-tense commit subjects (`Add VIF pruning report`).
- Keep commits focused; note any spec deviations in `docs/BUILD_LOG.md`.

## Project structure at a glance

| Path | Responsibility |
|---|---|
| `src/data/` | ingestion (real + synthetic), calibration, PyTorch dataset |
| `src/features/` | feature engineering + selection diagnostics |
| `src/models/` | loss, metrics, U-Net, training loop, inference |
| `src/utils/` | logging, device/seed, geospatial helpers, spatial CV |
| `app/` | FastAPI service, drift monitor, Streamlit dashboard |
| `scripts/` | thin CLI entry points (argparse → `src` functions) |

## Reporting bugs / requesting features

Open an issue at
<https://github.com/jainiakhil/Bushfire_Early_Warning_System/issues> with, ideally,
a minimal repro using the synthetic generator.

By contributing you agree your contributions are licensed under the project's
[MIT License](LICENSE).
