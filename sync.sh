#!/bin/bash
set -euo pipefail

BRANCH="${1:-main}"

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "ERROR: not inside a git repository."
  exit 1
fi

if ! git show-ref --verify --quiet "refs/heads/${BRANCH}"; then
  echo "ERROR: local branch '${BRANCH}' does not exist."
  exit 1
fi

if ! git remote get-url origin >/dev/null 2>&1; then
  echo "ERROR: remote 'origin' is missing."
  exit 1
fi

if ! git remote get-url hf >/dev/null 2>&1; then
  echo "ERROR: remote 'hf' is missing."
  exit 1
fi

echo "[1/4] Push ${BRANCH} to origin"
git push origin "${BRANCH}:${BRANCH}"

ORIGIN_HASH="$(git ls-remote origin "refs/heads/${BRANCH}" | awk '{print $1}')"
if [[ -z "${ORIGIN_HASH}" ]]; then
  echo "ERROR: could not read origin/${BRANCH} hash."
  exit 1
fi

echo "[2/4] Mirror origin/${BRANCH} to hf/${BRANCH}"
if git push hf "${ORIGIN_HASH}:refs/heads/${BRANCH}" --force-with-lease=refs/heads/${BRANCH}; then
  HF_HASH="$(git ls-remote hf "refs/heads/${BRANCH}" | awk '{print $1}')"
else
  HF_REMOTE_URL="$(git remote get-url hf)"
  HF_SPACE="$(echo "${HF_REMOTE_URL}" | sed -E 's#.*spaces/##; s#\.git$##')"
  if [[ -z "${HF_SPACE}" || "${HF_SPACE}" == "${HF_REMOTE_URL}" ]]; then
    HF_SPACE="manasdutta04/clinicaltrialenv"
  fi
  HF_USER_DEFAULT="${HF_SPACE%%/*}"
  HF_USERNAME="${HF_USERNAME:-${HF_USER_DEFAULT}}"
  HF_TOKEN="${HF_TOKEN:-}"

  if [[ -z "${HF_TOKEN}" ]]; then
    echo "ERROR: ssh push to 'hf' failed and HF_TOKEN is not set."
    echo "Set token and retry: HF_TOKEN=hf_xxx ./sync.sh ${BRANCH}"
    exit 1
  fi

  HF_HTTPS_URL="https://${HF_USERNAME}:${HF_TOKEN}@huggingface.co/spaces/${HF_SPACE}"
  HF_OLD_HASH="$(git ls-remote "${HF_HTTPS_URL}" "refs/heads/${BRANCH}" | awk '{print $1}')"
  if [[ -n "${HF_OLD_HASH}" ]]; then
    git push "${HF_HTTPS_URL}" "${ORIGIN_HASH}:refs/heads/${BRANCH}" "--force-with-lease=refs/heads/${BRANCH}:${HF_OLD_HASH}"
  else
    git push "${HF_HTTPS_URL}" "${ORIGIN_HASH}:refs/heads/${BRANCH}" --force
  fi
  HF_HASH="$(git ls-remote "${HF_HTTPS_URL}" "refs/heads/${BRANCH}" | awk '{print $1}')"
fi

if [[ -z "${HF_HASH}" ]]; then
  echo "ERROR: could not read hf/${BRANCH} hash."
  exit 1
fi

echo "[3/4] Verify remote parity"
echo "origin/${BRANCH}: ${ORIGIN_HASH}"
echo "hf/${BRANCH}:     ${HF_HASH}"

if [[ "${ORIGIN_HASH}" != "${HF_HASH}" ]]; then
  echo "ERROR: remotes differ after sync."
  exit 1
fi

echo "[4/4] Sync complete: remotes are identical"