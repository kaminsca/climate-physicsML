#!/bin/bash

# Check if the directory path is provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <directory-path>"
    exit 1
fi

# Directory path
DIR_PATH="$1"
# Output file
OUTPUT_FILE="all_python_files_content.txt"

# List of filenames (without directory paths) to exclude
EXCLUDE_FILES=("predict_and_visualize.py" "skip_this_one.py" "example.py")

# Clear the output file if it exists
> "$OUTPUT_FILE"

# Iterate over all .py files in the directory
find "$DIR_PATH" -type f -name "*.py" | while read -r file; do
    # Get the basename of the file
    BASENAME=$(basename "$file")
    
    # Check if the file is in the exclusion list
    if [[ " ${EXCLUDE_FILES[@]} " =~ " ${BASENAME} " ]]; then
        # Add the basename and exclusion message
        echo "###FILENAME: $file" >> "$OUTPUT_FILE"
        echo "### excluded because it is not important for the current matter ###" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"  # Add an empty line for readability
        continue
    fi

    # Append the filename header to the output file
    echo "###FILENAME: $file" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"  # Add an empty line for readability

    # Append the file content to the output file
    cat "$file" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"  # Add an empty line after file content for readability
done

echo "Contents of all .py files have been written to $OUTPUT_FILE with exclusions noted."
