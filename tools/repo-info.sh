#!/bin/bash
# Script to compare the installed sizes of pandas+numpy and lontras in a temporary virtual environment.
#
# This script performs the following steps:
# 1. Creates a temporary directory.
# 2. Creates and activates a virtual environment within the temporary directory.
# 3. Installs pandas and lontras in the virtual environment (silently).
# 4. Calculates time to import libraries
# 5. Calculates the installed size of pandas+numpy and lontras.
# 6. Formats the output in human-readable format.
# 7. Prints the results.
# 8. Cleans up the temporary directory upon exit.
#
# Functions:
#   get_package_size <package_name>: Returns the size of the installed package in bytes or "Not Installed".
#   format_output <size_bytes>: Formats the size in bytes to a human-readable format (e.g., 1.2M, 10K) or returns "Not Available".
#   setup: Sets up the temporary environment (creates directory, venv, installs packages).
#   compare_repo_size: Main function to perform the size comparison.

get_package_size() {
    local package
    local package_path
    package="$1"
    package_path="$(python -c "import site; print(site.getsitepackages()[0])")/$package"

    if [[ -d $package_path ]]; then
        du -s "$package_path" | cut -f1
    else
        echo "Not Installed"
        return 1
    fi
}

library_version() {
    local library
    library="$1"

    python -c "import $library as l;print(l.__version__)"
}

time_library_import() {
    local library
    local number_of_runs
    local total_time
    local time
    local average_time

    library="$1"
    number_of_runs="$2"
    total_time=0
    for _i in $(seq 1 "$number_of_runs"); do
        time=$(python -X importtime -c "import $library" 2>&1 | rg "[^\.]$library$" | cut -d "|" -f 2)
        if [[ -z $time ]]; then
            echo "Error: Could not find import time for $library" >&2
            return 1
        fi
        total_time=$(echo "$total_time + $time" | bc)
    done
    average_time=$(echo "scale=9; $total_time / $number_of_runs / 1000" | bc)
    echo "$average_time"
}

format_output() {
    local size="$1"
    if [[ $size == "Not Installed" ]]; then
        echo "Not Available"
    else
        numfmt --to=iec --from-unit=k --format="%1.1f" "$size"
    fi
}

setup() {
    tmp_dir=$(mktemp -d)
    trap 'rm -rf "$tmp_dir"' EXIT
    cd "$tmp_dir" || exit 1
    python3 -m venv .venv
    source .venv/bin/activate
    pip install pandas lontras >/dev/null 2>&1
}

compare_repo_size() {
    setup

    num_runs=20
    lontras_import_time=$(time_library_import lontras "$num_runs")
    pandas_import_time=$(time_library_import pandas "$num_runs")

    lontras_version=$(library_version lontras)
    pandas_version=$(library_version pandas)

    pandas_size_bytes=$(get_package_size pandas)
    numpy_size_bytes=$(get_package_size numpy)
    lontras_size_bytes=$(get_package_size lontras)

    lontras_size=$(format_output "$lontras_size_bytes")
    pandas_numpy_size=$(format_output $((pandas_size_bytes + numpy_size_bytes)))

    python -c "
import json
data = {
    'lontras': {'size': '$lontras_size', 'version': '$lontras_version', 'import-time': '$(printf "%.0f" "$lontras_import_time")ms'},
    'pandas': {'size': '$pandas_numpy_size', 'version': '$pandas_version', 'import-time': '$(printf "%.0f" "$pandas_import_time")ms'}
}
print(json.dumps(data, indent=2))
"
}

compare_repo_size
