#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Run the first Python script to process sources
echo "Running OpenAPI__process_sources.py..."
python OpenAPI__process_sources.py

# After it completes, run the main Python script
echo "Running debate_bot.py..."
python debate_bot.py