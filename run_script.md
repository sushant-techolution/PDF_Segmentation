## Batch (run every PDF in Data/ automatically)

python3 run_batch.py \
  --input-dir Data \
  --output-root output \
  --creds service-account.json \
  --project proposal-auto-ai-internal \
  --location us-central1 --gemini2-location us \
  --workers 5 \
  --taxonomy pss_taxonomy.json --prototypes pss_prototypes.json

Each PDF gets its own output folder named after its full filename, e.g.
output/20151425114_1016185945_20241212_v1/ — re-running the same PDF bumps
to _v2, _v3, etc. Add --skip-existing to resume a batch without re-running
PDFs that already have a _v1 folder. A batch_run_log_<timestamp>.csv summary
(status + timing per PDF) is written to --output-root when the batch finishes.

## Single PDF (manual, one file at a time)

python3 "Pss pipeline .py" "20151425114_1016185945_20241212.PDF" \
  --creds service-account.json \
  --project proposal-auto-ai-internal \
  --location us-central1 --gemini2-location us \
  --output-dir ./output/20151425114_1016185945_20241212_v1 --workers 5 \
  --taxonomy pss_taxonomy.json --prototypes pss_prototypes.json

<!-- python "Pss pipeline .py" "20171403887_1016186898_20241218.PDF" --creds service-account.json --project proposal-auto-ai-internal --location us-central1 --gemini2-location us --output-dir ./output_pdf1218_v1 --workers 5 --taxonomy pss_taxonomy.json --prototypes pss_prototypes.json -->

python3 run_batch.py --project proposal-auto-ai-internal --workers 5
python3 compare_to_ground_truth.py  ground_truth.xlsx  script_output.xlsx  -o result.xlsx