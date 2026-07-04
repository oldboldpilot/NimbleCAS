#!/bin/bash
# Backup script for NimbleCAS to NAS
# Olumuyiwa Oluwasanmi
#
# The NAS is a PRIVATE, full working-state backup, so config/.env IS included here
# (it holds credentials). config/.env must still NEVER be committed/pushed to the git
# remotes (github/gitea/mgpu) — it is .gitignored for exactly that reason. Only the
# git path excludes it; the NAS mirror keeps it.

SOURCE_DIR="/mnt/c/ClaudeCodeExec/NimbleCAS"
BACKUP_DIR="/mnt/z/PycharmProjects/NimbleCAS"

echo "Starting backup to NAS..."
echo "Source: $SOURCE_DIR"
echo "Destination: $BACKUP_DIR"

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Sync files using rsync (excludes build artifacts, cache, logs, symlinks)
rsync -rv --delete --no-g --no-o --no-p --no-t --no-links --ignore-errors --inplace \
    --exclude='build/' \
    --exclude='build-win/' \
    --exclude='.git/' \
    --exclude='*.log' \
    --exclude='*.pcm' \
    --exclude='*.obj' \
    --exclude='*.o' \
    --exclude='*.exe' \
    --exclude='*.dll' \
    --exclude='*.so' \
    --exclude='.venv/' \
    --exclude='.av_cache/' \
    --exclude='.claude/' \
    --exclude='.gemini/' \
    "$SOURCE_DIR/" "$BACKUP_DIR/"

if [ $? -eq 0 ]; then
    echo "✓ Backup completed successfully"
    echo "Backup location: $BACKUP_DIR"
else
    echo "✗ Backup failed with error code $?"
    exit 1
fi
