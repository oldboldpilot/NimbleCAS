#!/bin/bash
# Backup script for NimbleCAS to NAS
# Olumuyiwa Oluwasanmi

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
    --exclude='config/.env' \
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
