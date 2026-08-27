#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMMIT="7887401b09e25f54a546a253c255741cb1ab48e5"
VERSION="0.99.25"
OUT="${1:-$ROOT/publication_release_0.99.25/documented}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$OUT"
git -C "$ROOT" archive "$COMMIT" | tar -x -C "$TMP"
(cd "$TMP" && R CMD build .)
ARCHIVE="$(find "$TMP" -maxdepth 1 -name "fastPLS_${VERSION}.tar.gz" -print -quit)"
test -n "$ARCHIVE"
cp "$ARCHIVE" "$OUT/"

ARCHIVE_SHA="$(shasum -a 256 "$OUT/fastPLS_${VERSION}.tar.gz" | awk '{print $1}')"
{
  printf 'field\tvalue\n'
  printf 'artifact_role\tdocumented source archive\n'
  printf 'package\tfastPLS\n'
  printf 'version\t%s\n' "$VERSION"
  printf 'git_commit\t%s\n' "$COMMIT"
  printf 'archive\tfastPLS_%s.tar.gz\n' "$VERSION"
  printf 'archive_sha256\t%s\n' "$ARCHIVE_SHA"
  printf 'execution_archive_sha256\t%s\n' \
    '604f74d72f5e9540f7378efd37f2ca6ed87ccb9e73b912211331a8cd97233481'
  printf 'source_identity\tidentical Git commit; generated documentation differs\n'
} > "$OUT/documented_archive_manifest.tsv"
printf '%s  %s\n' "$ARCHIVE_SHA" "fastPLS_${VERSION}.tar.gz" > "$OUT/SHA256SUMS"
echo "$OUT/fastPLS_${VERSION}.tar.gz"
