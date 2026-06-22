#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 OUT_ROOT [OUT_ROOT ...]" >&2
  exit 2
fi

encode_stream() {
  local pattern=$1
  local out_file=$2
  local label=$3
  local tmp_file="${out_file%.mp4}.tmp.mp4"

  if ! compgen -G "${pattern}" >/dev/null; then
    return 0
  fi
  if [[ -s "${out_file}" ]]; then
    return 0
  fi

  mkdir -p -- "$(dirname "${out_file}")"
  echo "[encode] ${label} -> ${out_file}"
  rm -f -- "${tmp_file}"
  if ffmpeg -nostdin -hide_banner -loglevel error -threads 1 -y -framerate 10 -pattern_type glob \
    -i "${pattern}" \
    -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -pix_fmt yuv420p \
    "${tmp_file}"; then
    mv -f -- "${tmp_file}" "${out_file}"
    return 0
  fi

  rm -f -- "${tmp_file}"
  return 1
}

stream_exists() {
  local pattern=$1
  local out_file=$2
  compgen -G "${pattern}" >/dev/null && [[ ! -s "${out_file}" ]]
}

count_missing_streams() {
  local out_root=$1
  local total=0
  local exp_dir

  while IFS= read -r exp_dir; do
    stream_exists "${exp_dir}/camera_rgb/frame_*_agentview.png" "${exp_dir}/videos/agentview.mp4" && total=$((total + 1))
    stream_exists "${exp_dir}/camera_rgb/frame_*_eye.png" "${exp_dir}/videos/eye.mp4" && total=$((total + 1))
    stream_exists "${exp_dir}/tactile_markers_rgb/frame_*_gsmini_left_markers_rgb.png" "${exp_dir}/videos/gsmini_left_markers_rgb.mp4" && total=$((total + 1))
    stream_exists "${exp_dir}/tactile_markers_rgb/frame_*_gsmini_right_markers_rgb.png" "${exp_dir}/videos/gsmini_right_markers_rgb.mp4" && total=$((total + 1))
  done < <(find "${out_root}/debug" -type d -name 'exp_*' 2>/dev/null | sort)

  echo "${total}"
}

backfill_one_root() {
  local out_root=$1
  local log_dir="${out_root}/logs"
  local stamp
  stamp=$(date +%Y%m%d_%H%M%S)
  local log_file="${log_dir}/backfill_video_encode_${stamp}.log"
  local total=0
  local index=0
  local ok=0
  local fail=0
  local exp_dir

  if [[ ! -d "${out_root}/debug" ]]; then
    echo "[skip] missing debug dir: ${out_root}" >&2
    return 0
  fi

  mkdir -p -- "${log_dir}"
  total=$(count_missing_streams "${out_root}")
  {
    echo "START $(date -Iseconds) out_root=${out_root} missing=${total}"
    if [[ "${total}" -eq 0 ]]; then
      echo "DONE $(date -Iseconds) ok=0 fail=0"
      return 0
    fi

    while IFS= read -r exp_dir; do
      for spec in \
        "camera_rgb/frame_*_agentview.png|videos/agentview.mp4|agentview" \
        "camera_rgb/frame_*_eye.png|videos/eye.mp4|eye" \
        "tactile_markers_rgb/frame_*_gsmini_left_markers_rgb.png|videos/gsmini_left_markers_rgb.mp4|gsmini_left_markers_rgb" \
        "tactile_markers_rgb/frame_*_gsmini_right_markers_rgb.png|videos/gsmini_right_markers_rgb.mp4|gsmini_right_markers_rgb"; do
        local rel_pattern=${spec%%|*}
        local rest=${spec#*|}
        local rel_out=${rest%%|*}
        local label=${rest##*|}
        local pattern="${exp_dir}/${rel_pattern}"
        local out_file="${exp_dir}/${rel_out}"

        if ! stream_exists "${pattern}" "${out_file}"; then
          continue
        fi

        index=$((index + 1))
        echo "[${index}/${total}] ${out_file#${out_root}/}"
        if encode_stream "${pattern}" "${out_file}" "${label}"; then
          ok=$((ok + 1))
        else
          fail=$((fail + 1))
          echo "[fail] ${out_file#${out_root}/}"
        fi
      done
    done < <(find "${out_root}/debug" -type d -name 'exp_*' 2>/dev/null | sort)

    echo "DONE $(date -Iseconds) ok=${ok} fail=${fail}"
  } >> "${log_file}" 2>&1
}

for out_root in "$@"; do
  backfill_one_root "${out_root}"
done
