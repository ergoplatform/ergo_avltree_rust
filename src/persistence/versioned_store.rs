//! Versioned key-value store backed by `redb`.
//!
//! Mirrors the design of Scala's `LDBVersionedStore` from the Ergo node,
//! using an undo-log pattern for rollback support.
//!
//! # Architecture
//!
//! Four strictly-typed `redb` tables:
//! - `NODES_TABLE`:    hash → serialized node bytes
//! - `META_TABLE`:     string key → global state (last_version, next_lsn)
//! - `UNDO_LOG_TABLE`: LSN → undo entry bytes
//! - `VERSIONS_TABLE`: version_id → version record bytes

use alloc::vec::Vec;
use anyhow::{anyhow, Result};
use redb::{Database, ReadableTable, TableDefinition};
use std::path::Path;

/// Main data table: node hash (32 bytes) → serialized node
const NODES_TABLE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("nodes");
/// Global metadata: string keys → byte values
const META_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("meta");
/// Undo log: monotonic LSN → serialized UndoEntry
const UNDO_LOG_TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("undo_log");
/// Version registry: version_id bytes → serialized VersionRecord
const VERSIONS_TABLE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("versions");

// Meta keys
const META_LAST_VERSION: &str = "last_version";
const META_NEXT_LSN: &str = "next_lsn";

/// A single undo operation: "to undo this write, either delete or restore"
struct UndoEntry {
    key: Vec<u8>,
    /// None = key didn't exist before (undo = delete it)
    /// Some(v) = key had this value before (undo = restore it)
    old_value: Option<Vec<u8>>,
}

impl UndoEntry {
    fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        // key_len (u32) + key + has_old (u8) + [old_len (u32) + old_value]
        buf.extend_from_slice(&(self.key.len() as u32).to_le_bytes());
        buf.extend_from_slice(&self.key);
        match &self.old_value {
            None => buf.push(0u8),
            Some(v) => {
                buf.push(1u8);
                buf.extend_from_slice(&(v.len() as u32).to_le_bytes());
                buf.extend_from_slice(v);
            }
        }
        buf
    }

    fn deserialize(data: &[u8]) -> Result<UndoEntry> {
        let mut pos = 0;
        if data.len() < 4 {
            return Err(anyhow!("UndoEntry too short"));
        }
        let key_len = u32::from_le_bytes(data[pos..pos + 4].try_into()?) as usize;
        pos += 4;
        if data.len() < pos + key_len + 1 {
            return Err(anyhow!("UndoEntry key truncated"));
        }
        let key = data[pos..pos + key_len].to_vec();
        pos += key_len;
        let has_old = data[pos];
        pos += 1;
        let old_value = if has_old == 1 {
            if data.len() < pos + 4 {
                return Err(anyhow!("UndoEntry old_value length truncated"));
            }
            let old_len = u32::from_le_bytes(data[pos..pos + 4].try_into()?) as usize;
            pos += 4;
            if data.len() < pos + old_len {
                return Err(anyhow!("UndoEntry old_value truncated"));
            }
            Some(data[pos..pos + old_len].to_vec())
        } else {
            None
        };
        Ok(UndoEntry { key, old_value })
    }
}

/// Metadata for a version checkpoint, forming a linked list for rollback traversal.
struct VersionRecord {
    /// Hash digest identifying this version
    version_id: Vec<u8>,
    /// Previous version (linked list for rollback traversal)
    parent_version_id: Option<Vec<u8>>,
    /// First undo LSN for this version (inclusive)
    start_lsn: u64,
    /// Last undo LSN for this version (inclusive)
    end_lsn: u64,
}

impl VersionRecord {
    fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        // version_id_len + version_id
        buf.extend_from_slice(&(self.version_id.len() as u32).to_le_bytes());
        buf.extend_from_slice(&self.version_id);
        // has_parent + [parent_len + parent]
        match &self.parent_version_id {
            None => buf.push(0u8),
            Some(p) => {
                buf.push(1u8);
                buf.extend_from_slice(&(p.len() as u32).to_le_bytes());
                buf.extend_from_slice(p);
            }
        }
        // start_lsn + end_lsn
        buf.extend_from_slice(&self.start_lsn.to_le_bytes());
        buf.extend_from_slice(&self.end_lsn.to_le_bytes());
        buf
    }

    fn deserialize(data: &[u8]) -> Result<VersionRecord> {
        let mut pos = 0;
        // version_id
        let vid_len = u32::from_le_bytes(data[pos..pos + 4].try_into()?) as usize;
        pos += 4;
        let version_id = data[pos..pos + vid_len].to_vec();
        pos += vid_len;
        // parent
        let has_parent = data[pos];
        pos += 1;
        let parent_version_id = if has_parent == 1 {
            let p_len = u32::from_le_bytes(data[pos..pos + 4].try_into()?) as usize;
            pos += 4;
            let p = data[pos..pos + p_len].to_vec();
            pos += p_len;
            Some(p)
        } else {
            None
        };
        // LSNs
        let start_lsn = u64::from_le_bytes(data[pos..pos + 8].try_into()?);
        pos += 8;
        let end_lsn = u64::from_le_bytes(data[pos..pos + 8].try_into()?);
        Ok(VersionRecord {
            version_id,
            parent_version_id,
            start_lsn,
            end_lsn,
        })
    }
}

/// A versioned key-value store backed by `redb` with undo-log rollback.
///
/// This mirrors Scala's `LDBVersionedStore`:
/// - `update()` atomically applies changes and records compensating undo entries
/// - `rollback()` walks the undo log backwards from the current version to the target
pub struct RedbVersionedStore {
    db: Database,
}

impl RedbVersionedStore {
    /// Open or create a versioned store at the given path.
    pub fn open(path: &Path) -> Result<Self> {
        let db = Database::create(path)?;

        // Ensure all tables exist
        let txn = db.begin_write()?;
        {
            let _ = txn.open_table(NODES_TABLE)?;
            let _ = txn.open_table(META_TABLE)?;
            let _ = txn.open_table(UNDO_LOG_TABLE)?;
            let _ = txn.open_table(VERSIONS_TABLE)?;
        }
        txn.commit()?;

        Ok(RedbVersionedStore { db })
    }

    /// Get the last version ID, or None if store is empty.
    pub fn last_version_id(&self) -> Result<Option<Vec<u8>>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(META_TABLE)?;
        match table.get(META_LAST_VERSION)? {
            Some(v) => Ok(Some(v.value().to_vec())),
            None => Ok(None),
        }
    }

    /// Read a node by its hash key.
    pub fn get_node(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(NODES_TABLE)?;
        match table.get(key)? {
            Some(v) => Ok(Some(v.value().to_vec())),
            None => Ok(None),
        }
    }

    /// Atomically apply an update: insert/update/remove keys in NODES_TABLE,
    /// and record undo entries for rollback.
    ///
    /// - `version_id`: the digest identifying this version
    /// - `to_insert`: keys and values to insert or update
    /// - `to_remove`: keys to remove
    pub fn update(
        &self,
        version_id: &[u8],
        to_insert: &[(&[u8], &[u8])],
        to_remove: &[&[u8]],
    ) -> Result<()> {
        let txn = self.db.begin_write()?;
        {
            let mut nodes = txn.open_table(NODES_TABLE)?;
            let mut undo_log = txn.open_table(UNDO_LOG_TABLE)?;
            let mut meta = txn.open_table(META_TABLE)?;
            let mut versions = txn.open_table(VERSIONS_TABLE)?;

            // Read next_lsn and parent version INSIDE the write txn (Q4 TOCTOU fix)
            let mut next_lsn = meta
                .get(META_NEXT_LSN)?
                .map(|v| {
                    let bytes: [u8; 8] = v.value().try_into().unwrap();
                    u64::from_le_bytes(bytes)
                })
                .unwrap_or(0);
            let start_lsn = next_lsn;
            let parent_version = meta.get(META_LAST_VERSION)?.map(|v| v.value().to_vec());

            // Record undo entries for removals
            for key in to_remove {
                let old_value = nodes.get(*key)?.map(|v| v.value().to_vec());
                if old_value.is_some() {
                    let undo = UndoEntry {
                        key: key.to_vec(),
                        old_value,
                    };
                    undo_log.insert(next_lsn, undo.serialize().as_slice())?;
                    next_lsn += 1;
                    nodes.remove(*key)?;
                }
            }

            // Record undo entries for inserts/updates
            for (key, value) in to_insert {
                let old_value = nodes.get(*key)?.map(|v| v.value().to_vec());
                let undo = UndoEntry {
                    key: key.to_vec(),
                    old_value,
                };
                undo_log.insert(next_lsn, undo.serialize().as_slice())?;
                next_lsn += 1;
                nodes.insert(*key, *value)?;
            }

            // Record version
            let end_lsn = if next_lsn > start_lsn {
                next_lsn - 1
            } else {
                start_lsn
            };
            let record = VersionRecord {
                version_id: version_id.to_vec(),
                parent_version_id: parent_version,
                start_lsn,
                end_lsn,
            };
            versions.insert(version_id, record.serialize().as_slice())?;

            // Update meta
            meta.insert(META_LAST_VERSION, version_id)?;
            meta.insert(META_NEXT_LSN, next_lsn.to_le_bytes().as_slice())?;
        }
        txn.commit()?;
        Ok(())
    }

    /// Rollback to a target version by walking the undo log backwards
    /// through the version linked list.
    pub fn rollback(&self, target_version: &[u8]) -> Result<()> {
        // Walk from current version back to target
        let current_version = self
            .last_version_id()?
            .ok_or_else(|| anyhow!("Store is empty, cannot rollback"))?;

        if current_version == target_version {
            return Ok(()); // Already at target
        }

        // Collect versions to undo (from current back to target, exclusive of target)
        let mut versions_to_undo = Vec::new();
        let mut cursor = current_version.clone();

        let read_txn = self.db.begin_read()?;
        let versions_table = read_txn.open_table(VERSIONS_TABLE)?;

        loop {
            if cursor == target_version {
                break;
            }
            let record_bytes = versions_table
                .get(cursor.as_slice())?
                .ok_or_else(|| anyhow!("Version not found in chain: {:?}", &cursor[..4]))?;
            let record = VersionRecord::deserialize(record_bytes.value())?;
            versions_to_undo.push((record.start_lsn, record.end_lsn, record.version_id.clone()));
            cursor = record
                .parent_version_id
                .ok_or_else(|| anyhow!("Reached genesis without finding target version"))?;
        }
        drop(versions_table);
        drop(read_txn);

        // Apply undo entries in reverse order
        let txn = self.db.begin_write()?;
        {
            let mut nodes = txn.open_table(NODES_TABLE)?;
            let mut undo_log = txn.open_table(UNDO_LOG_TABLE)?;
            let mut meta = txn.open_table(META_TABLE)?;
            let mut versions = txn.open_table(VERSIONS_TABLE)?;

            for (start_lsn, end_lsn, version_id) in &versions_to_undo {
                // Apply undo entries in reverse LSN order
                let mut lsn = *end_lsn;
                loop {
                    let entry_data = undo_log.get(lsn)?.map(|guard| guard.value().to_vec());
                    if let Some(data) = entry_data {
                        let entry = UndoEntry::deserialize(&data)?;
                        match entry.old_value {
                            None => {
                                // Key didn't exist before → delete it
                                nodes.remove(entry.key.as_slice())?;
                            }
                            Some(old_val) => {
                                // Key had a previous value → restore it
                                nodes.insert(entry.key.as_slice(), old_val.as_slice())?;
                            }
                        }
                        undo_log.remove(lsn)?;
                    }
                    if lsn == *start_lsn {
                        break;
                    }
                    lsn -= 1;
                }
                // Remove the version record
                versions.remove(version_id.as_slice())?;
            }

            // Update meta to point to the target version
            meta.insert(META_LAST_VERSION, target_version)?;

            // Recalculate next_lsn from the target version
            let target_data = versions
                .get(target_version)?
                .map(|guard| guard.value().to_vec());
            if let Some(data) = target_data {
                let target_record = VersionRecord::deserialize(&data)?;
                let new_next_lsn = target_record.end_lsn + 1;
                meta.insert(META_NEXT_LSN, new_next_lsn.to_le_bytes().as_slice())?;
            }
        }
        txn.commit()?;
        Ok(())
    }

    /// Force a durable sync to disk.
    pub fn flush(&self) -> Result<()> {
        // redb commits are already durable after txn.commit()
        // This is provided for trait compatibility.
        Ok(())
    }

    /// Return all version IDs in reverse chronological order.
    pub fn rollback_versions(&self) -> Result<Vec<Vec<u8>>> {
        let mut versions = Vec::new();
        let mut cursor = self.last_version_id()?;

        let txn = self.db.begin_read()?;
        let versions_table = txn.open_table(VERSIONS_TABLE)?;

        while let Some(vid) = cursor {
            versions.push(vid.clone());
            match versions_table.get(vid.as_slice())? {
                Some(record_bytes) => {
                    let record = VersionRecord::deserialize(record_bytes.value())?;
                    cursor = record.parent_version_id;
                }
                None => break,
            }
        }
        Ok(versions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_basic_insert_and_read() {
        let dir = tempdir().unwrap();
        let store = RedbVersionedStore::open(&dir.path().join("test.redb")).unwrap();

        // Insert some data
        store
            .update(
                b"version_1",
                &[(b"key_a".as_slice(), b"value_a".as_slice())],
                &[],
            )
            .unwrap();

        // Read it back
        let val = store.get_node(b"key_a").unwrap();
        assert_eq!(val, Some(b"value_a".to_vec()));

        // Check version
        assert_eq!(
            store.last_version_id().unwrap(),
            Some(b"version_1".to_vec())
        );
    }

    #[test]
    fn test_update_and_rollback() {
        let dir = tempdir().unwrap();
        let store = RedbVersionedStore::open(&dir.path().join("test.redb")).unwrap();

        // Version 1: insert key_a
        store
            .update(b"v1", &[(b"key_a".as_slice(), b"val_1".as_slice())], &[])
            .unwrap();

        // Version 2: update key_a, insert key_b
        store
            .update(
                b"v2",
                &[
                    (b"key_a".as_slice(), b"val_2".as_slice()),
                    (b"key_b".as_slice(), b"val_b".as_slice()),
                ],
                &[],
            )
            .unwrap();

        // Verify current state
        assert_eq!(store.get_node(b"key_a").unwrap(), Some(b"val_2".to_vec()));
        assert_eq!(store.get_node(b"key_b").unwrap(), Some(b"val_b".to_vec()));

        // Rollback to v1
        store.rollback(b"v1").unwrap();

        // key_a should be back to val_1
        assert_eq!(store.get_node(b"key_a").unwrap(), Some(b"val_1".to_vec()));
        // key_b should be gone (didn't exist in v1)
        assert_eq!(store.get_node(b"key_b").unwrap(), None);
        // Version should be v1
        assert_eq!(store.last_version_id().unwrap(), Some(b"v1".to_vec()));
    }

    #[test]
    fn test_multi_version_rollback() {
        let dir = tempdir().unwrap();
        let store = RedbVersionedStore::open(&dir.path().join("test.redb")).unwrap();

        // Create 5 versions
        for i in 0..5u8 {
            let ver = [b'v', i + b'0'];
            let key = [b'k', i + b'0'];
            let val = [b'v', b'a', b'l', i + b'0'];
            store.update(&ver, &[(&key[..], &val[..])], &[]).unwrap();
        }

        // Rollback to v2
        store.rollback(b"v2").unwrap();

        // Keys k0, k1, k2 should exist; k3, k4 should not
        assert!(store.get_node(b"k0").unwrap().is_some());
        assert!(store.get_node(b"k1").unwrap().is_some());
        assert!(store.get_node(b"k2").unwrap().is_some());
        assert!(store.get_node(b"k3").unwrap().is_none());
        assert!(store.get_node(b"k4").unwrap().is_none());
    }

    #[test]
    fn test_removal_with_undo() {
        let dir = tempdir().unwrap();
        let store = RedbVersionedStore::open(&dir.path().join("test.redb")).unwrap();

        // v1: insert key
        store
            .update(b"v1", &[(b"mykey".as_slice(), b"myval".as_slice())], &[])
            .unwrap();

        // v2: remove key
        store.update(b"v2", &[], &[b"mykey".as_slice()]).unwrap();
        assert_eq!(store.get_node(b"mykey").unwrap(), None);

        // Rollback to v1: key should be restored
        store.rollback(b"v1").unwrap();
        assert_eq!(store.get_node(b"mykey").unwrap(), Some(b"myval".to_vec()));
    }

    #[test]
    fn test_rollback_versions_list() {
        let dir = tempdir().unwrap();
        let store = RedbVersionedStore::open(&dir.path().join("test.redb")).unwrap();

        store
            .update(b"v1", &[(b"k".as_slice(), b"1".as_slice())], &[])
            .unwrap();
        store
            .update(b"v2", &[(b"k".as_slice(), b"2".as_slice())], &[])
            .unwrap();
        store
            .update(b"v3", &[(b"k".as_slice(), b"3".as_slice())], &[])
            .unwrap();

        let versions = store.rollback_versions().unwrap();
        assert_eq!(versions.len(), 3);
        assert_eq!(versions[0], b"v3");
        assert_eq!(versions[1], b"v2");
        assert_eq!(versions[2], b"v1");
    }
}
