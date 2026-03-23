python -m pip install -r requirements.txt
#python scripts/data-download/download-raw-data.py
python scripts/data-normalization/flights-normalization.py
python scripts/data-normalization/airports-normalization.py
python scripts/data-normalization/airlines-normalization.py
python scripts/data-download/build-curated-parquet.py
python scripts/feature-engineering/build_features.py
python scripts/modeling/flights-modeling.py