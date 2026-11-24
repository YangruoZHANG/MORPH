# Verification

The seven supplied metric tests passed with Python 3.13.9. The stored-result validator passed for 360 root rows, 270 dataset rows, nine methods and 30 datasets. All seven supplied result hashes matched the source archive. Full scoring from raw predictions was not run because that external input directory is absent from the archive. The original method_id values retain compatibility with that input directory.

Commands executed from this directory

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider
python scripts/validate.py
```
