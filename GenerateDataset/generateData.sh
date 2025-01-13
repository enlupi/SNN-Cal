#!/bin/bash

# Check if the script receives at least one input parameter
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 <end> [<start>]"
    exit 1
fi

# Validate that the input parameters are positive integers
if ! [[ $1 =~ ^[0-9]+$ ]]; then
    echo "Error: Ending file number must be a positive integer."
    exit 1
fi

if [ "$#" -eq 2 ] && ! [[ $2 =~ ^[0-9]+$ ]]; then
    echo "Error: Starting file number must be a positive integer."
    exit 1
fi

# Get the start and end values
if [ "$#" -eq 2 ]; then
    start=$1
    end=$2
else
    start=1
    end=$1
fi

# Ensure start is less than or equal to end
if [ "$start" -gt "$end" ]; then
    echo "Error: Starting file number cannot be greater than ending number."
    exit 1
fi

# Array of particle types
particles=("kaon" "proton" "pion")

# Loop over each particle type and file index
for particle in "${particles[@]}"; do
    for ((i=start; i<=end; i++)); do
        input_file="/lustre/cmsdata/optCalData/${particle}/${particle}_${i}.root"
        
        # Check if the input file exists
        if [ -f "$input_file" ]; then
            echo "Processing file: $input_file"
            ./genPhotons -f "$input_file" -o ../Data/PrimaryOnly/Centered/${particle} -po -r 0 -v 1
        else
            echo "Warning: File not found - $input_file"
        fi
    done
done
