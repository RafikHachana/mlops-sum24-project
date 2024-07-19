#!/bin/bash

python3 src/sample_and_validate.py

# Check the exit status of the command
if [ $? -ne 0 ]; then
  echo "Validation failed, exiting."
  exit 1
else
  echo "Validation succeeded, committing to the repo."
fi

dvc add data/samples/sample.csv
git add data/samples/sample.csv.dvc
git commit -m "New data sample"
