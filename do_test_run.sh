#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="example_algorithm_debug"

DOCKER_NOOP_VOLUME="${DOCKER_IMAGE_TAG}-volume"

INPUT_DIR="${SCRIPT_DIR}/test/input"
OUTPUT_DIR="${SCRIPT_DIR}/test/outputs"

echo "=+= (Re)build the container"
source "${SCRIPT_DIR}/do_build.sh"

cleanup() {
    echo "=+= Cleaning permissions ..."
    # Ensure permissions are set correctly on the output
    # This allows the host user (e.g. you) to access and handle these files
    docker run --rm \
      --platform=linux/amd64 \
      --quiet \
      --volume "$OUTPUT_DIR":/output \
      --entrypoint /bin/sh \
      $DOCKER_IMAGE_TAG \
      -c "chmod -R -f o+rwX /output/* || true"

    # Ensure volume is removed
    docker volume rm "$DOCKER_NOOP_VOLUME" > /dev/null
}

# This allows for the Docker user to read (best-effort: on some mounted
# filesystems the host user does not own these files and chmod is not permitted,
# so never let it abort the run under `set -e`).
chmod -R -f o+rX "$INPUT_DIR" "${SCRIPT_DIR}/model" || true

# Each case is a directory holding an inputs.json (e.g. interf0/case1). Grand
# Challenge runs one job per case, so we mirror that: one container run per
# case, each seeing a flat /input and writing its result sockets to /output.
mapfile -t CASE_DIRS < <(cd "$INPUT_DIR" && find . -name inputs.json -printf '%h\n' | sed 's|^\./||' | sort)
if [ "${#CASE_DIRS[@]}" -eq 0 ]; then
  echo "=+= No cases (inputs.json) found under ${INPUT_DIR}" >&2
  exit 1
fi

# Fresh output tree (clean via the container to sidestep ownership problems).
if [ -d "$OUTPUT_DIR" ]; then
  chmod -R -f o+rwX "$OUTPUT_DIR" || true

  echo "=+= Cleaning up any earlier output"
  docker run --rm \
      --platform=linux/amd64 \
      --quiet \
      --volume "$OUTPUT_DIR":/output \
      --entrypoint /bin/sh \
      $DOCKER_IMAGE_TAG \
      -c "rm -rf /output/* || true"
else
  mkdir -p -m o+rwX "$OUTPUT_DIR"
fi


docker volume create "$DOCKER_NOOP_VOLUME" > /dev/null

trap cleanup EXIT

run_docker_forward_pass() {
    local case_dir="$1"

    echo "=+= Doing a forward pass on ${case_dir}"

    # Per-case output dir must exist and be writable by the container user.
    mkdir -p -m o+rwX "${OUTPUT_DIR}/${case_dir}"

    ## Note the extra arguments that are passed here:
    # '--network none'
    #    entails there is no internet connection
    # '--gpus all'
    #    enables access to any GPUs present
    # '--volume <NAME>:/tmp'
    #   is added because on Grand Challenge this directory cannot be used to store permanent files
    # '--volume ../model:/opt/ml/model/":ro'
    #   is added to provide access to the (optional) tarball-upload locally
  docker run --rm --gpus all \
      --platform=linux/amd64 \
      --network none \
      --volume "${INPUT_DIR}/${case_dir}":/input:ro \
      --volume "${OUTPUT_DIR}/${case_dir}":/output \
      --volume "$DOCKER_NOOP_VOLUME":/tmp \
      --volume "${SCRIPT_DIR}/model":/opt/ml/model:ro \
      "$DOCKER_IMAGE_TAG"

  echo "=+= Wrote results to ${OUTPUT_DIR}/${case_dir}"
}


for case_dir in "${CASE_DIRS[@]}"; do
  run_docker_forward_pass "$case_dir"
done

# Make the container-written outputs host-readable, then rebuild the combined
# predictions.json exactly as the Grand Challenge platform would across jobs.
docker run --rm \
    --platform=linux/amd64 \
    --quiet \
    --volume "$OUTPUT_DIR":/output \
    --entrypoint /bin/sh \
    $DOCKER_IMAGE_TAG \
    -c "chmod -R -f o+rwX /output/* || true"

echo "=+= Aggregating predictions.json"
# Run the aggregator with the container's Python (the host python3 may predate
# 3.7); it reads /input + the per-case /output sockets and writes predictions.json.
docker run --rm \
    --platform=linux/amd64 \
    --quiet \
    --volume "${SCRIPT_DIR}/scripts":/opt/app/scripts:ro \
    --volume "${INPUT_DIR}":/input:ro \
    --volume "${OUTPUT_DIR}":/output \
    --entrypoint python3 \
    "$DOCKER_IMAGE_TAG" \
    /opt/app/scripts/aggregate_predictions.py --input /input --output /output


echo "=+= Save this image for uploading via ./do_save.sh"
