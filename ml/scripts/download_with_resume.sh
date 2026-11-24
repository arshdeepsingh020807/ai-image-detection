#!/usr/bin/env bash
# Usage: download_with_resume.sh <kaggle-slug> <target-dir> [keep-zip]
# Resumes via Kaggle's partial-file support; restarts the CLI when the zip stops growing for 60 s.
# With "keep-zip" the archive is left unextracted (for selective extraction on small disks).
set -u
slug="$1"; dir="$2"; keep_zip="${3:-}"; mkdir -p "$dir"
zip="$dir/${slug#*/}.zip"
unzip_flag="--unzip"; [ "$keep_zip" = "keep-zip" ] && unzip_flag=""
until [ -e "$dir/.complete" ]; do
  kaggle datasets download -d "$slug" $unzip_flag -p "$dir" & pid=$!
  last=-1
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
    size=$(stat -f %z "$zip" 2>/dev/null || echo 0)
    if [ "$size" = "$last" ] && kill -0 "$pid" 2>/dev/null; then echo "stalled at $size, restarting"; kill "$pid"; break; fi
    last=$size
  done
  wait "$pid" 2>/dev/null && touch "$dir/.complete"
done
