#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

### CONFIGURATION ###
BASE_PATH="$HOME/workspaces/oi"
CAM_DIR="${BASE_PATH}/cams/HikVision"
OUT_DIR="${BASE_PATH}/output"
DB_SCRIPT="tmp_create.sql"
DB_NAME="intelligence"
CONTAINER_NAME="postgres"

### FUNCTIONS ###
log() {
    printf "[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

safe_delete() {
    local target="$1"
    if [[ -d "$target" ]]; then
        log "Deleting files in: $target"
        find "$target" -type f -print -delete
    else
        log "ERROR: Directory not found: $target"
        exit 1
    fi
}

### MAIN SCRIPT ###
log "Bringing docker-compose services down..."
docker-compose down

safe_delete "$CAM_DIR"
safe_delete "$OUT_DIR"

log "Creating DB init script..."
cat > "$DB_SCRIPT" <<EOF
DROP DATABASE IF EXISTS $DB_NAME;
CREATE DATABASE $DB_NAME;
EOF

log "Copying script into container..."
docker cp "$DB_SCRIPT" "$CONTAINER_NAME:/tmp/$DB_SCRIPT"

sleep 2

log "Executing DB script inside container..."
docker exec -u postgres "$CONTAINER_NAME" psql postgres postgres -f "/tmp/$DB_SCRIPT"

log "Cleaning up local temp script..."
rm -f "$DB_SCRIPT"

log "Done."
exit 0