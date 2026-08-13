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

use alloc::{collections::BTreeSet, vec::Vec};
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

struct DecodeCursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> DecodeCursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn take(&mut self, len: usize, field: &'static str) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(len)
            .ok_or_else(|| anyhow!("{} length overflow", field))?;
        let value = self
            .data
            .get(self.pos..end)
            .ok_or_else(|| anyhow!("{} truncated", field))?;
        self.pos = end;
        Ok(value)
    }

    fn read_u8(&mut self, field: &'static str) -> Result<u8> {
        Ok(self.take(1, field)?[0])
    }

    fn read_u32_le(&mut self, field: &'static str) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4, field)?.try_into()?))
    }

    fn read_u64_le(&mut self, field: &'static str) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8, field)?.try_into()?))
    }

    fn finish(self, kind: &'static str) -> Result<()> {
        if self.pos != self.data.len() {
            return Err(anyhow!("{} has trailing bytes", kind));
        }
        Ok(())
    }
}

impl UndoEntry {
    fn serialize(&self) -> Result<Vec<u8>> {
        let mut buf = Vec::new();
        // key_len (u32) + key + has_old (u8) + [old_len (u32) + old_value]
        let key_len = u32::try_from(self.key.len())
            .map_err(|_| anyhow!("UndoEntry key length exceeds u32"))?;
        buf.extend_from_slice(&key_len.to_le_bytes());
        buf.extend_from_slice(&self.key);
        match &self.old_value {
            None => buf.push(0u8),
            Some(v) => {
                buf.push(1u8);
                let old_len = u32::try_from(v.len())
                    .map_err(|_| anyhow!("UndoEntry old value length exceeds u32"))?;
                buf.extend_from_slice(&old_len.to_le_bytes());
                buf.extend_from_slice(v);
            }
        }
        Ok(buf)
    }

    fn deserialize(data: &[u8]) -> Result<UndoEntry> {
        let mut cursor = DecodeCursor::new(data);
        let key_len = usize::try_from(cursor.read_u32_le("UndoEntry key length")?)?;
        let key = cursor.take(key_len, "UndoEntry key")?.to_vec();
        let old_value = match cursor.read_u8("UndoEntry old-value tag")? {
            0 => None,
            1 => {
                let old_len = usize::try_from(cursor.read_u32_le("UndoEntry old-value length")?)?;
                Some(cursor.take(old_len, "UndoEntry old value")?.to_vec())
            }
            _ => return Err(anyhow!("UndoEntry old-value tag must be 0 or 1")),
        };
        cursor.finish("UndoEntry")?;
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
    fn serialize(&self) -> Result<Vec<u8>> {
        let mut buf = Vec::new();
        // version_id_len + version_id
        let version_len = u32::try_from(self.version_id.len())
            .map_err(|_| anyhow!("VersionRecord version ID length exceeds u32"))?;
        buf.extend_from_slice(&version_len.to_le_bytes());
        buf.extend_from_slice(&self.version_id);
        // has_parent + [parent_len + parent]
        match &self.parent_version_id {
            None => buf.push(0u8),
            Some(p) => {
                buf.push(1u8);
                let parent_len = u32::try_from(p.len())
                    .map_err(|_| anyhow!("VersionRecord parent ID length exceeds u32"))?;
                buf.extend_from_slice(&parent_len.to_le_bytes());
                buf.extend_from_slice(p);
            }
        }
        // start_lsn + end_lsn
        buf.extend_from_slice(&self.start_lsn.to_le_bytes());
        buf.extend_from_slice(&self.end_lsn.to_le_bytes());
        Ok(buf)
    }

    fn deserialize(data: &[u8]) -> Result<VersionRecord> {
        let mut cursor = DecodeCursor::new(data);
        let version_len = usize::try_from(cursor.read_u32_le("VersionRecord version ID length")?)?;
        let version_id = cursor
            .take(version_len, "VersionRecord version ID")?
            .to_vec();
        let parent_version_id = match cursor.read_u8("VersionRecord parent tag")? {
            0 => None,
            1 => {
                let parent_len =
                    usize::try_from(cursor.read_u32_le("VersionRecord parent ID length")?)?;
                Some(cursor.take(parent_len, "VersionRecord parent ID")?.to_vec())
            }
            _ => return Err(anyhow!("VersionRecord parent tag must be 0 or 1")),
        };
        let start_lsn = cursor.read_u64_le("VersionRecord start LSN")?;
        let end_lsn = cursor.read_u64_le("VersionRecord end LSN")?;
        cursor.finish("VersionRecord")?;
        if start_lsn > end_lsn {
            return Err(anyhow!("VersionRecord has an invalid LSN range"));
        }
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

    #[cfg(test)]
    pub(crate) fn replace_node_for_test(&self, key: &[u8], value: &[u8]) -> Result<()> {
        let txn = self.db.begin_write()?;
        {
            let mut nodes = txn.open_table(NODES_TABLE)?;
            nodes.insert(key, value)?;
        }
        txn.commit()?;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn remove_version_for_test(&self, version_id: &[u8]) -> Result<()> {
        let txn = self.db.begin_write()?;
        {
            let mut versions = txn.open_table(VERSIONS_TABLE)?;
            versions.remove(version_id)?;
        }
        txn.commit()?;
        Ok(())
    }

    /// Verify that a repeated AVL digest describes the exact current state.
    ///
    /// The low-level update API always rejects reused version IDs. The AVL
    /// adapter may use this read-only check to coalesce a proof-only update
    /// whose digest and persisted content are unchanged.
    pub(crate) fn verify_current_state(
        &self,
        version_id: &[u8],
        expected_entries: &[(&[u8], &[u8])],
        expected_absent: &[&[u8]],
    ) -> Result<()> {
        let txn = self.db.begin_read()?;
        let nodes = txn.open_table(NODES_TABLE)?;
        let meta = txn.open_table(META_TABLE)?;
        let versions = txn.open_table(VERSIONS_TABLE)?;

        let current = meta
            .get(META_LAST_VERSION)?
            .map(|value| value.value().to_vec());
        if current.as_deref() != Some(version_id) {
            return Err(anyhow!("Version ID is not the current persistence tip"));
        }
        let record_bytes = versions
            .get(version_id)?
            .ok_or_else(|| anyhow!("Current persistence version record is missing"))?;
        let record = VersionRecord::deserialize(record_bytes.value())?;
        if record.version_id != version_id {
            return Err(anyhow!("Version record ID does not match lookup key"));
        }

        for (key, expected_value) in expected_entries {
            let stored = nodes.get(*key)?.map(|value| value.value().to_vec());
            if stored.as_deref() != Some(*expected_value) {
                return Err(anyhow!(
                    "Repeated AVL digest does not match persisted state"
                ));
            }
        }
        for key in expected_absent {
            if nodes.get(*key)?.is_some() {
                return Err(anyhow!(
                    "Repeated AVL digest conflicts with a persisted removal"
                ));
            }
        }

        Ok(())
    }

    /// Atomically apply ordinary upserts and removals.
    pub fn update(
        &self,
        version_id: &[u8],
        to_insert: &[(&[u8], &[u8])],
        to_remove: &[&[u8]],
    ) -> Result<()> {
        self.update_partitioned(version_id, &[], to_insert, to_remove)
    }

    /// Atomically apply content-addressed inserts, ordinary upserts, and removals.
    ///
    /// Existing content-addressed keys must contain the exact proposed bytes.
    /// The comparison and any subsequent mutation share one Redb write transaction.
    pub(crate) fn update_partitioned(
        &self,
        version_id: &[u8],
        content_addressed: &[(&[u8], &[u8])],
        upserts: &[(&[u8], &[u8])],
        to_remove: &[&[u8]],
    ) -> Result<()> {
        let txn = self.db.begin_write()?;
        let update_result = (|| -> Result<()> {
            let mut nodes = txn.open_table(NODES_TABLE)?;
            let mut undo_log = txn.open_table(UNDO_LOG_TABLE)?;
            let mut meta = txn.open_table(META_TABLE)?;
            let mut versions = txn.open_table(VERSIONS_TABLE)?;

            if versions.get(version_id)?.is_some() {
                return Err(anyhow!("Version ID already exists in history"));
            }

            for (key, value) in content_addressed {
                if let Some(stored) = nodes.get(*key)? {
                    if stored.value() != *value {
                        return Err(anyhow!(
                            "Content-addressed node key conflicts with different stored bytes"
                        ));
                    }
                }
            }

            // Read next_lsn and parent version INSIDE the write txn (Q4 TOCTOU fix)
            let mut next_lsn = match meta.get(META_NEXT_LSN)? {
                Some(value) => {
                    let bytes: [u8; 8] = value
                        .value()
                        .try_into()
                        .map_err(|_| anyhow!("Persisted next LSN must be exactly 8 bytes"))?;
                    u64::from_le_bytes(bytes)
                }
                None => 0,
            };
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
                    let encoded = undo.serialize()?;
                    undo_log.insert(next_lsn, encoded.as_slice())?;
                    next_lsn = next_lsn
                        .checked_add(1)
                        .ok_or_else(|| anyhow!("Undo-log LSN overflow"))?;
                    nodes.remove(*key)?;
                }
            }

            // Insert content only when it is absent after removals. Recheck the
            // exact bytes so duplicate keys in this batch cannot disagree.
            for (key, value) in content_addressed {
                let stored = {
                    let guard = nodes.get(*key)?;
                    guard.map(|value| value.value().to_vec())
                };
                match stored {
                    Some(stored) => {
                        if stored.as_slice() != *value {
                            return Err(anyhow!(
                                "Content-addressed node key conflicts with different stored bytes"
                            ));
                        }
                    }
                    None => {
                        let undo = UndoEntry {
                            key: key.to_vec(),
                            old_value: None,
                        };
                        let encoded = undo.serialize()?;
                        undo_log.insert(next_lsn, encoded.as_slice())?;
                        next_lsn = next_lsn
                            .checked_add(1)
                            .ok_or_else(|| anyhow!("Undo-log LSN overflow"))?;
                        nodes.insert(*key, *value)?;
                    }
                }
            }

            // Metadata and other ordinary entries are always upserted.
            for (key, value) in upserts {
                let old_value = nodes.get(*key)?.map(|v| v.value().to_vec());
                let undo = UndoEntry {
                    key: key.to_vec(),
                    old_value,
                };
                let encoded = undo.serialize()?;
                undo_log.insert(next_lsn, encoded.as_slice())?;
                next_lsn = next_lsn
                    .checked_add(1)
                    .ok_or_else(|| anyhow!("Undo-log LSN overflow"))?;
                nodes.insert(*key, *value)?;
            }

            if next_lsn == start_lsn {
                return Err(anyhow!(
                    "Version update must contain at least one node mutation"
                ));
            }

            let end_lsn = next_lsn
                .checked_sub(1)
                .ok_or_else(|| anyhow!("Undo-log LSN underflow"))?;
            let record = VersionRecord {
                version_id: version_id.to_vec(),
                parent_version_id: parent_version,
                start_lsn,
                end_lsn,
            };
            let encoded_record = record.serialize()?;
            versions.insert(version_id, encoded_record.as_slice())?;

            // Update meta
            meta.insert(META_LAST_VERSION, version_id)?;
            meta.insert(META_NEXT_LSN, next_lsn.to_le_bytes().as_slice())?;
            Ok(())
        })();

        match update_result {
            Ok(()) => {
                txn.commit()?;
                Ok(())
            }
            Err(error) => {
                if let Err(abort_error) = txn.abort() {
                    return Err(anyhow!(
                        "{}; failed to abort Redb update transaction: {}",
                        error,
                        abort_error
                    ));
                }
                Err(error)
            }
        }
    }

    /// Rollback to a target version by walking the undo log backwards.
    pub fn rollback(&self, target_version: &[u8]) -> Result<()> {
        self.rollback_with_validation(target_version, |_| Ok(()))
    }

    /// Roll back and validate the uncommitted target state before committing it.
    pub(crate) fn rollback_with_validation<T>(
        &self,
        target_version: &[u8],
        validate: impl FnOnce(&dyn Fn(&[u8]) -> Result<Option<Vec<u8>>>) -> Result<T>,
    ) -> Result<T> {
        let txn = self.db.begin_write()?;
        let rollback_result = (|| -> Result<T> {
            let mut nodes = txn.open_table(NODES_TABLE)?;
            let mut undo_log = txn.open_table(UNDO_LOG_TABLE)?;
            let mut meta = txn.open_table(META_TABLE)?;
            let mut versions = txn.open_table(VERSIONS_TABLE)?;

            let current_version = meta
                .get(META_LAST_VERSION)?
                .map(|value| value.value().to_vec())
                .ok_or_else(|| anyhow!("Store is empty, cannot rollback"))?;
            let current_next_lsn = match meta.get(META_NEXT_LSN)? {
                Some(value) => {
                    let bytes: [u8; 8] = value
                        .value()
                        .try_into()
                        .map_err(|_| anyhow!("Persisted next LSN must be exactly 8 bytes"))?;
                    u64::from_le_bytes(bytes)
                }
                None => return Err(anyhow!("Persisted next LSN is missing")),
            };

            let target_data = versions
                .get(target_version)?
                .map(|value| value.value().to_vec())
                .ok_or_else(|| anyhow!("Target version record is missing"))?;
            let target_record = VersionRecord::deserialize(&target_data)?;
            if target_record.version_id != target_version {
                return Err(anyhow!("Version record ID does not match lookup key"));
            }

            let mut versions_to_undo = Vec::new();
            let mut cursor = current_version;
            let mut visited = BTreeSet::new();
            let mut expected_next_lsn = current_next_lsn;
            while cursor != target_version {
                if !visited.insert(cursor.clone()) {
                    return Err(anyhow!("Cycle detected in version history"));
                }
                let record_data = versions
                    .get(cursor.as_slice())?
                    .map(|value| value.value().to_vec())
                    .ok_or_else(|| anyhow!("Version record is missing from rollback chain"))?;
                let record = VersionRecord::deserialize(&record_data)?;
                if record.version_id != cursor {
                    return Err(anyhow!("Version record ID does not match lookup key"));
                }
                let record_next_lsn = record
                    .end_lsn
                    .checked_add(1)
                    .ok_or_else(|| anyhow!("Undo-log LSN overflow"))?;
                if record_next_lsn != expected_next_lsn {
                    return Err(anyhow!("Version undo-log ranges are not contiguous"));
                }
                expected_next_lsn = record.start_lsn;
                cursor = record
                    .parent_version_id
                    .clone()
                    .ok_or_else(|| anyhow!("Reached genesis without finding target version"))?;
                versions_to_undo.push(record);
            }

            let target_next_lsn = target_record
                .end_lsn
                .checked_add(1)
                .ok_or_else(|| anyhow!("Undo-log LSN overflow"))?;
            if target_next_lsn != expected_next_lsn {
                return Err(anyhow!("Target version undo-log range is not contiguous"));
            }

            for record in &versions_to_undo {
                let mut lsn = record.end_lsn;
                loop {
                    let entry_data = undo_log
                        .get(lsn)?
                        .map(|guard| guard.value().to_vec())
                        .ok_or_else(|| anyhow!("Undo-log entry is missing"))?;
                    let entry = UndoEntry::deserialize(&entry_data)?;
                    match entry.old_value {
                        None => {
                            nodes.remove(entry.key.as_slice())?;
                        }
                        Some(old_value) => {
                            nodes.insert(entry.key.as_slice(), old_value.as_slice())?;
                        }
                    }
                    if undo_log.remove(lsn)?.is_none() {
                        return Err(anyhow!("Undo-log entry disappeared during rollback"));
                    }
                    if lsn == record.start_lsn {
                        break;
                    }
                    lsn = lsn
                        .checked_sub(1)
                        .ok_or_else(|| anyhow!("Undo-log LSN underflow"))?;
                }
                if versions.remove(record.version_id.as_slice())?.is_none() {
                    return Err(anyhow!("Version record disappeared during rollback"));
                }
            }

            meta.insert(META_LAST_VERSION, target_version)?;
            meta.insert(META_NEXT_LSN, target_next_lsn.to_le_bytes().as_slice())?;

            let lookup = |key: &[u8]| -> Result<Option<Vec<u8>>> {
                Ok(nodes.get(key)?.map(|value| value.value().to_vec()))
            };
            validate(&lookup)
        })();

        match rollback_result {
            Ok(value) => {
                txn.commit()?;
                Ok(value)
            }
            Err(error) => {
                if let Err(abort_error) = txn.abort() {
                    return Err(anyhow!(
                        "{}; failed to abort Redb rollback transaction: {}",
                        error,
                        abort_error
                    ));
                }
                Err(error)
            }
        }
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
        let txn = self.db.begin_read()?;
        let versions_table = txn.open_table(VERSIONS_TABLE)?;
        let meta = txn.open_table(META_TABLE)?;
        let mut cursor = meta
            .get(META_LAST_VERSION)?
            .map(|value| value.value().to_vec());
        let mut visited = BTreeSet::new();

        while let Some(vid) = cursor {
            if !visited.insert(vid.clone()) {
                return Err(anyhow!("Cycle detected in version history"));
            }
            match versions_table.get(vid.as_slice())? {
                Some(record_bytes) => {
                    let record = VersionRecord::deserialize(record_bytes.value())?;
                    if record.version_id != vid {
                        return Err(anyhow!("Version record ID does not match lookup key"));
                    }
                    versions.push(vid);
                    cursor = record.parent_version_id;
                }
                None => return Err(anyhow!("Version record is missing from history")),
            }
        }
        Ok(versions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use tempfile::tempdir;

    fn assert_decode_err_without_panic<T>(decode: impl FnOnce() -> Result<T>) {
        let outcome = catch_unwind(AssertUnwindSafe(decode));
        assert!(outcome.is_ok(), "malformed persisted bytes must not panic");
        assert!(
            outcome.unwrap().is_err(),
            "malformed persisted bytes must fail"
        );
    }

    #[test]
    fn undo_entry_decoder_rejects_every_truncation_invalid_tags_and_trailing_bytes() {
        for entry in [
            UndoEntry {
                key: b"key".to_vec(),
                old_value: None,
            },
            UndoEntry {
                key: b"key".to_vec(),
                old_value: Some(b"old".to_vec()),
            },
        ] {
            let encoded = entry.serialize().unwrap();
            for end in 0..encoded.len() {
                assert_decode_err_without_panic(|| UndoEntry::deserialize(&encoded[..end]));
            }

            let mut trailing = encoded.clone();
            trailing.push(0);
            assert_decode_err_without_panic(|| UndoEntry::deserialize(&trailing));
        }

        for tag in [2, 255] {
            let malformed = [1, 0, 0, 0, b'k', tag];
            assert_decode_err_without_panic(|| UndoEntry::deserialize(&malformed));
        }
    }

    #[test]
    fn version_record_decoder_rejects_every_truncation_invalid_tags_ranges_and_trailing_bytes() {
        for record in [
            VersionRecord {
                version_id: b"v1".to_vec(),
                parent_version_id: None,
                start_lsn: 0,
                end_lsn: 0,
            },
            VersionRecord {
                version_id: b"v2".to_vec(),
                parent_version_id: Some(b"v1".to_vec()),
                start_lsn: 1,
                end_lsn: 2,
            },
        ] {
            let encoded = record.serialize().unwrap();
            for end in 0..encoded.len() {
                assert_decode_err_without_panic(|| VersionRecord::deserialize(&encoded[..end]));
            }

            let mut trailing = encoded.clone();
            trailing.push(0);
            assert_decode_err_without_panic(|| VersionRecord::deserialize(&trailing));
        }

        for tag in [2, 255] {
            let mut malformed = Vec::new();
            malformed.extend_from_slice(&2u32.to_le_bytes());
            malformed.extend_from_slice(b"v1");
            malformed.push(tag);
            malformed.extend_from_slice(&0u64.to_le_bytes());
            malformed.extend_from_slice(&0u64.to_le_bytes());
            assert_decode_err_without_panic(|| VersionRecord::deserialize(&malformed));
        }

        let invalid_range = VersionRecord {
            version_id: b"v1".to_vec(),
            parent_version_id: None,
            start_lsn: 2,
            end_lsn: 1,
        }
        .serialize()
        .unwrap();
        assert_decode_err_without_panic(|| VersionRecord::deserialize(&invalid_range));
    }

    #[test]
    fn malformed_next_lsn_is_an_error_not_a_panic() {
        let dir = tempdir().unwrap();
        let store = RedbVersionedStore::open(&dir.path().join("test.redb")).unwrap();
        let txn = store.db.begin_write().unwrap();
        {
            let mut meta = txn.open_table(META_TABLE).unwrap();
            meta.insert(META_NEXT_LSN, &[0u8; 7][..]).unwrap();
        }
        txn.commit().unwrap();

        let outcome = catch_unwind(AssertUnwindSafe(|| {
            store.update(b"v1", &[(b"key", b"value")], &[])
        }));
        assert!(outcome.is_ok(), "malformed next_lsn must not panic");
        assert!(outcome.unwrap().is_err(), "malformed next_lsn must fail");
        assert_eq!(store.last_version_id().unwrap(), None);
        assert_eq!(store.get_node(b"key").unwrap(), None);
    }

    #[test]
    fn update_without_any_node_mutation_is_rejected() {
        let dir = tempdir().unwrap();
        let store = RedbVersionedStore::open(&dir.path().join("test.redb")).unwrap();

        let error = store.update(b"v1", &[], &[]).unwrap_err();

        assert_eq!(
            error.to_string(),
            "Version update must contain at least one node mutation"
        );
        assert_eq!(store.last_version_id().unwrap(), None);
        assert!(store.rollback_versions().unwrap().is_empty());
    }

    fn version_record(store: &RedbVersionedStore, version_id: &[u8]) -> VersionRecord {
        let txn = store.db.begin_read().unwrap();
        let versions = txn.open_table(VERSIONS_TABLE).unwrap();
        let bytes = versions.get(version_id).unwrap().unwrap().value().to_vec();
        VersionRecord::deserialize(&bytes).unwrap()
    }

    fn replace_version_record(
        store: &RedbVersionedStore,
        version_id: &[u8],
        record: &VersionRecord,
    ) {
        let txn = store.db.begin_write().unwrap();
        {
            let mut versions = txn.open_table(VERSIONS_TABLE).unwrap();
            versions
                .insert(version_id, record.serialize().unwrap().as_slice())
                .unwrap();
        }
        txn.commit().unwrap();
    }

    fn remove_version_record(store: &RedbVersionedStore, version_id: &[u8]) {
        let txn = store.db.begin_write().unwrap();
        {
            let mut versions = txn.open_table(VERSIONS_TABLE).unwrap();
            versions.remove(version_id).unwrap();
        }
        txn.commit().unwrap();
    }

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
    fn rollback_validation_error_aborts_and_preserves_undo_history() {
        let dir = tempdir().unwrap();
        let store = RedbVersionedStore::open(&dir.path().join("test.redb")).unwrap();
        store.update(b"v0", &[(b"key", b"zero")], &[]).unwrap();
        store.update(b"v1", &[(b"key", b"one")], &[]).unwrap();

        let error = store
            .rollback_with_validation(b"v0", |lookup| {
                assert_eq!(lookup(b"key")?, Some(b"zero".to_vec()));
                Err::<(), _>(anyhow!("reject candidate"))
            })
            .unwrap_err();

        assert_eq!(error.to_string(), "reject candidate");
        assert_eq!(store.last_version_id().unwrap(), Some(b"v1".to_vec()));
        assert_eq!(store.get_node(b"key").unwrap(), Some(b"one".to_vec()));
        assert_eq!(
            store.rollback_versions().unwrap(),
            vec![b"v1".to_vec(), b"v0".to_vec()]
        );

        store.rollback(b"v0").unwrap();
        assert_eq!(store.get_node(b"key").unwrap(), Some(b"zero".to_vec()));
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

    #[test]
    fn rollback_versions_rejects_a_missing_history_record() {
        let dir = tempdir().unwrap();
        let store = RedbVersionedStore::open(&dir.path().join("test.redb")).unwrap();
        store.update(b"v1", &[(b"key", b"one")], &[]).unwrap();
        store.update(b"v2", &[(b"key", b"two")], &[]).unwrap();
        remove_version_record(&store, b"v1");

        let error = store.rollback_versions().unwrap_err();

        assert_eq!(error.to_string(), "Version record is missing from history");
    }

    #[test]
    fn test_duplicate_tip_version_id_is_rejected_atomically() {
        let dir = tempdir().unwrap();
        let store = RedbVersionedStore::open(&dir.path().join("test.redb")).unwrap();

        store
            .update(b"v1", &[(b"key".as_slice(), b"value".as_slice())], &[])
            .unwrap();
        let error = store
            .update(b"v1", &[(b"key".as_slice(), b"value".as_slice())], &[])
            .unwrap_err();

        assert_eq!(error.to_string(), "Version ID already exists in history");
        assert_eq!(store.get_node(b"key").unwrap(), Some(b"value".to_vec()));
        assert_eq!(store.last_version_id().unwrap(), Some(b"v1".to_vec()));
        assert_eq!(version_record(&store, b"v1").parent_version_id, None);
        assert_eq!(store.rollback_versions().unwrap(), vec![b"v1".to_vec()]);
    }

    #[test]
    fn test_duplicate_tip_conflict_is_rejected_atomically() {
        let dir = tempdir().unwrap();
        let store = RedbVersionedStore::open(&dir.path().join("test.redb")).unwrap();

        store
            .update(b"v1", &[(b"key".as_slice(), b"value".as_slice())], &[])
            .unwrap();

        let error = store
            .update(b"v1", &[(b"key".as_slice(), b"different".as_slice())], &[])
            .unwrap_err();

        assert_eq!(error.to_string(), "Version ID already exists in history");
        assert_eq!(store.get_node(b"key").unwrap(), Some(b"value".to_vec()));
        assert_eq!(store.last_version_id().unwrap(), Some(b"v1".to_vec()));
        assert_eq!(store.rollback_versions().unwrap(), vec![b"v1".to_vec()]);
    }

    #[test]
    fn test_historical_version_recurrence_is_rejected_atomically() {
        let dir = tempdir().unwrap();
        let store = RedbVersionedStore::open(&dir.path().join("test.redb")).unwrap();

        store
            .update(b"v0", &[(b"key".as_slice(), b"zero".as_slice())], &[])
            .unwrap();
        store
            .update(b"v1", &[(b"key".as_slice(), b"one".as_slice())], &[])
            .unwrap();

        let error = store
            .update(b"v0", &[(b"key".as_slice(), b"zero".as_slice())], &[])
            .unwrap_err();

        assert_eq!(error.to_string(), "Version ID already exists in history");
        assert_eq!(store.get_node(b"key").unwrap(), Some(b"one".to_vec()));
        assert_eq!(store.last_version_id().unwrap(), Some(b"v1".to_vec()));
        assert_eq!(
            store.rollback_versions().unwrap(),
            vec![b"v1".to_vec(), b"v0".to_vec()]
        );
        store.rollback(b"v0").unwrap();
        assert_eq!(store.get_node(b"key").unwrap(), Some(b"zero".to_vec()));
    }

    #[test]
    fn test_rollback_rejects_version_record_key_mismatch() {
        let dir = tempdir().unwrap();
        let store = RedbVersionedStore::open(&dir.path().join("test.redb")).unwrap();

        store.update(b"v0", &[(b"key", b"zero")], &[]).unwrap();
        store.update(b"v1", &[(b"key", b"one")], &[]).unwrap();

        let mut record = version_record(&store, b"v1");
        record.version_id = b"wrong".to_vec();
        replace_version_record(&store, b"v1", &record);

        let error = store.rollback(b"v0").unwrap_err();
        assert_eq!(
            error.to_string(),
            "Version record ID does not match lookup key"
        );
        assert_eq!(store.last_version_id().unwrap(), Some(b"v1".to_vec()));
        assert_eq!(store.get_node(b"key").unwrap(), Some(b"one".to_vec()));
    }

    #[test]
    fn test_rollback_versions_rejects_cycle() {
        let dir = tempdir().unwrap();
        let store = RedbVersionedStore::open(&dir.path().join("test.redb")).unwrap();

        store.update(b"v0", &[(b"key", b"zero")], &[]).unwrap();
        store.update(b"v1", &[(b"key", b"one")], &[]).unwrap();

        let mut record = version_record(&store, b"v0");
        record.parent_version_id = Some(b"v1".to_vec());
        replace_version_record(&store, b"v0", &record);

        let error = store.rollback_versions().unwrap_err();
        assert_eq!(error.to_string(), "Cycle detected in version history");
    }
}
