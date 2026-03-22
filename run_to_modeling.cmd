python -m pip install -r requirements.txt
python scripts/data-normalization/flights-normalization.py
python scripts/data-download/script-create-parquet-structure.py
python scripts/feature-engineering/build_features.py