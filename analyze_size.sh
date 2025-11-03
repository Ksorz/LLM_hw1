#!/bin/bash

# Print header
printf "%-30s %10s %15s %15s  %s\n" "Folder" "Files" "Size (MB)" "Largest (MB)" "Largest File Path"
printf "%-30s %10s %15s %15s  %s\n" "$(printf '%.0s─' {1..30})" "$(printf '%.0s─' {1..10})" "$(printf '%.0s─' {1..15})" "$(printf '%.0s─' {1..15})" "$(printf '%.0s─' {1..50})"

# Initialize totals
total_files=0
total_size=0

# Analyze each directory
for dir in */; do
  [[ -d "$dir" ]] || continue
  
  folder_name="${dir%/}"
  num_files=$(find "$dir" -type f 2>/dev/null | wc -l)
  mb_folder_size=$(du -sm "$dir" 2>/dev/null | cut -f1)
  
  # Get both size and path of the heaviest file
  heaviest_info=$(find "$dir" -type f -exec du -m {} + 2>/dev/null | sort -rn | head -1)
  heaviest_mb=$(echo "$heaviest_info" | cut -f1)
  heaviest_path=$(echo "$heaviest_info" | cut -f2-)
  
  [[ -z "$heaviest_mb" ]] && heaviest_mb=0
  [[ -z "$heaviest_path" ]] && heaviest_path="-"

  printf "%-30s %10s %15s %15s  %s\n" \
    "$folder_name" "$num_files" "$mb_folder_size" "$heaviest_mb" "$heaviest_path"
  
  # Add to totals
  total_files=$((total_files + num_files))
  total_size=$((total_size + mb_folder_size))
done

# Print totals
printf "%-30s %10s %15s %15s  %s\n" "$(printf '%.0s─' {1..30})" "$(printf '%.0s─' {1..10})" "$(printf '%.0s─' {1..15})" "$(printf '%.0s─' {1..15})" ""
printf "%-30s %10s %15s %15s  %s\n" "TOTAL" "$total_files" "$total_size" "-" ""