python3 "Pss pipeline .py" "synthetic_test_file_02.PDF" \
  --creds service-account.json \
  --project proposal-auto-ai-internal \
  --location us-central1 --gemini2-location us \
  --output-dir ./output_pdf02_v1 --workers 5 \
  --taxonomy pss_taxonomy.json --prototypes pss_prototypes.json