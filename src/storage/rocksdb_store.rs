use crate::operation::ADDigest;
use crate::storage::error::StorageError;
use crate::storage::kv_store::{KVStore, Result, StorageBatch, VersionedKVStore};
use bytes::Bytes;
use parking_lot::RwLock;
use rocksdb::{ColumnFamilyDescriptor, Options, WriteBatch, DB};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;

const CF_MAIN: &str = "main";
const CF_UNDO: &str = "undo";
const CF_META: &str = "meta";

const KEY_CURRENT_VERSION: &[u8] = b"__current_version__";
const KEY_LSN_COUNTER: &[u8] = b"__lsn_counter__";
const KEY_ROOT_HASH: &[u8] = b"__root_hash__";
const KEY_ROOT_HEIGHT: &[u8] = b"__root_height__";

type LSN = u64;

#[derive(Serialize, Deserialize, Debug, Clone)]
struct UndoRecord {
    lsn: LSN,
    version_id: Vec<u8>,
    key: Vec<u8>,
    old_value: Option<Vec<u8>>, // None means key didn't exist before
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct VersionMetadata {
    lsn_start: LSN,
    lsn_end: LSN,
    version_id: Vec<u8>,
    timestamp: u64,
}

/// RocksDB-backed key-value store
pub struct RocksDBStore {
    db: Arc<RwLock<DB>>,
}

impl RocksDBStore {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);

        let cf_main = ColumnFamilyDescriptor::new(CF_MAIN, Options::default());
        let cf_undo = ColumnFamilyDescriptor::new(CF_UNDO, Options::default());
        let cf_meta = ColumnFamilyDescriptor::new(CF_META, Options::default());

        let db = DB::open_cf_descriptors(&opts, path, vec![cf_main, cf_undo, cf_meta])?;
        Ok(Self {
            db: Arc::new(RwLock::new(db)),
        })
    }

    fn get_cf(&self, cf_name: &str, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let db = self.db.read();
        let cf = db
            .cf_handle(cf_name)
            .ok_or_else(|| StorageError::DatabaseError(format!("CF {} not found", cf_name)))?;
        Ok(db.get_cf(cf, key)?)
    }

    fn put_cf(&mut self, cf_name: &str, key: Vec<u8>, value: Vec<u8>) -> Result<()> {
        let db = self.db.write();
        let cf = db
            .cf_handle(cf_name)
            .ok_or_else(|| StorageError::DatabaseError(format!("CF {} not found", cf_name)))?;
        Ok(db.put_cf(cf, key, value)?)
    }

    fn delete_cf(&mut self, cf_name: &str, key: &[u8]) -> Result<()> {
        let db = self.db.write();
        let cf = db
            .cf_handle(cf_name)
            .ok_or_else(|| StorageError::DatabaseError(format!("CF {} not found", cf_name)))?;
        Ok(db.delete_cf(cf, key)?)
    }
}

impl KVStore for RocksDBStore {
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.get_cf(CF_MAIN, key)
    }

    fn put(&mut self, key: Vec<u8>, value: Vec<u8>) -> Result<()> {
        self.put_cf(CF_MAIN, key, value)
    }

    fn delete(&mut self, key: &[u8]) -> Result<()> {
        self.delete_cf(CF_MAIN, key)
    }

    fn write_batch(&mut self, batch: StorageBatch) -> Result<()> {
        let db = self.db.write();
        let cf = db
            .cf_handle(CF_MAIN)
            .ok_or_else(|| StorageError::DatabaseError("CF main not found".to_string()))?;

        let mut wb = WriteBatch::default();
        for (key, value) in batch.puts {
            wb.put_cf(cf, key, value);
        }
        for key in batch.deletes {
            wb.delete_cf(cf, key);
        }
        Ok(db.write(wb)?)
    }
}

/// Versioned RocksDB store with rollback capability
pub struct VersionedRocksDBStore {
    store: RocksDBStore,
    keep_versions: usize,
    version_history: VecDeque<ADDigest>,
}

impl VersionedRocksDBStore {
    pub fn new<P: AsRef<Path>>(path: P, keep_versions: usize) -> Result<Self> {
        let store = RocksDBStore::new(path)?;
        let mut vstore = Self {
            store,
            keep_versions,
            version_history: VecDeque::new(),
        };

        // Load version history from metadata
        vstore.load_version_history()?;
        Ok(vstore)
    }

    fn load_version_history(&mut self) -> Result<()> {
        // Scan metadata CF for version records
        let db = self.store.db.read();
        let cf = db
            .cf_handle(CF_META)
            .ok_or_else(|| StorageError::DatabaseError("CF meta not found".to_string()))?;

        let iter = db.iterator_cf(cf, rocksdb::IteratorMode::Start);
        let mut versions: Vec<(u64, ADDigest)> = Vec::new();

        for item in iter {
            let (key, value) = item?;
            if key.starts_with(b"version_") {
                let meta: VersionMetadata = bincode::deserialize(&value)?;
                versions.push((meta.timestamp, Bytes::from(meta.version_id)));
            }
        }

        versions.sort_by_key(|(ts, _)| *ts);
        self.version_history = versions.into_iter().map(|(_, v)| v).collect();
        Ok(())
    }

    fn get_lsn_counter(&self) -> Result<LSN> {
        let db = self.store.db.read();
        let cf = db
            .cf_handle(CF_META)
            .ok_or_else(|| StorageError::DatabaseError("CF meta not found".to_string()))?;

        match db.get_cf(cf, KEY_LSN_COUNTER)? {
            Some(bytes) => {
                let lsn = u64::from_be_bytes(bytes.try_into().map_err(|_| {
                    StorageError::InvalidData("Invalid LSN counter".to_string())
                })?);
                Ok(lsn)
            }
            None => Ok(0),
        }
    }

    fn increment_lsn_counter(&mut self) -> Result<LSN> {
        let current = self.get_lsn_counter()?;
        let next = current + 1;

        let db = self.store.db.write();
        let cf = db
            .cf_handle(CF_META)
            .ok_or_else(|| StorageError::DatabaseError("CF meta not found".to_string()))?;
        db.put_cf(cf, KEY_LSN_COUNTER, next.to_be_bytes())?;
        Ok(next)
    }

    fn save_undo_record(&mut self, record: &UndoRecord) -> Result<()> {
        let db = self.store.db.write();
        let cf = db
            .cf_handle(CF_UNDO)
            .ok_or_else(|| StorageError::DatabaseError("CF undo not found".to_string()))?;

        // Use inverted LSN as key for descending order iteration
        let inverted_lsn = !record.lsn;
        let key = inverted_lsn.to_be_bytes();
        let value = bincode::serialize(record)?;
        db.put_cf(cf, key, value)?;
        Ok(())
    }

    fn get_undo_records_for_version(&self, target_version: &[u8]) -> Result<Vec<UndoRecord>> {
        let db = self.store.db.read();
        let cf = db
            .cf_handle(CF_UNDO)
            .ok_or_else(|| StorageError::DatabaseError("CF undo not found".to_string()))?;

        let mut records = Vec::new();
        let iter = db.iterator_cf(cf, rocksdb::IteratorMode::Start);

        for item in iter {
            let (_, value) = item?;
            let record: UndoRecord = bincode::deserialize(&value)?;
            
            if record.version_id == target_version {
                records.push(record);
            }
        }

        records.sort_by_key(|r| std::cmp::Reverse(r.lsn));
        Ok(records)
    }
}

impl KVStore for VersionedRocksDBStore {
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.store.get(key)
    }

    fn put(&mut self, key: Vec<u8>, value: Vec<u8>) -> Result<()> {
        self.store.put(key, value)
    }

    fn delete(&mut self, key: &[u8]) -> Result<()> {
        self.store.delete(key)
    }

    fn write_batch(&mut self, batch: StorageBatch) -> Result<()> {
        self.store.write_batch(batch)
    }
}

impl VersionedKVStore for VersionedRocksDBStore {
    fn update_version(
        &mut self,
        version: &ADDigest,
        to_insert: Vec<(Vec<u8>, Vec<u8>)>,
        to_remove: Vec<Vec<u8>>,
    ) -> Result<()> {
        let lsn_start = self.increment_lsn_counter()?;
        let db = self.store.db.write();
        let cf_main = db
            .cf_handle(CF_MAIN)
            .ok_or_else(|| StorageError::DatabaseError("CF main not found".to_string()))?;
        let cf_undo = db
            .cf_handle(CF_UNDO)
            .ok_or_else(|| StorageError::DatabaseError("CF undo not found".to_string()))?;
        let cf_meta = db
            .cf_handle(CF_META)
            .ok_or_else(|| StorageError::DatabaseError("CF meta not found".to_string()))?;

        let mut wb = WriteBatch::default();
        let mut lsn = lsn_start;

        // Process insertions and create undo records
        for (key, value) in to_insert {
            let old_value = db.get_cf(cf_main, &key)?;
            
            wb.put_cf(cf_main, &key, &value);

            let undo = UndoRecord {
                lsn,
                version_id: version.to_vec(),
                key: key.clone(),
                old_value,
            };
            let inverted_lsn = !lsn;
            wb.put_cf(cf_undo, inverted_lsn.to_be_bytes(), bincode::serialize(&undo)?);
            lsn += 1;
        }

        // Process deletions and create undo records
        for key in to_remove {
            let old_value = db.get_cf(cf_main, &key)?;
            
            if old_value.is_some() {
                wb.delete_cf(cf_main, &key);

                let undo = UndoRecord {
                    lsn,
                    version_id: version.to_vec(),
                    key: key.clone(),
                    old_value,
                };
                let inverted_lsn = !lsn;
                wb.put_cf(cf_undo, inverted_lsn.to_be_bytes(), bincode::serialize(&undo)?);
                lsn += 1;
            }
        }

        let lsn_end = lsn - 1;

        // Save version metadata
        let version_meta = VersionMetadata {
            lsn_start,
            lsn_end,
            version_id: version.to_vec(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };
        wb.put_cf(
            cf_meta,
            format!("version_{}", base16::encode_lower(&version)).as_bytes(),
            bincode::serialize(&version_meta)?,
        );

        // Update current version
        wb.put_cf(cf_meta, KEY_CURRENT_VERSION, version);

        db.write(wb)?;
        drop(db);

        // Update version history
        self.version_history.push_back(version.clone());
        if self.version_history.len() > self.keep_versions {
            self.version_history.pop_front();
        }

        Ok(())
    }

    fn rollback_to_version(&mut self, version: &ADDigest) -> Result<()> {
        if !self.version_history.contains(version) {
            return Err(StorageError::VersionNotFound(format!(
                "Version {} not found",
                base16::encode_lower(version)
            )));
        }

        let current = self.current_version().ok_or_else(|| {
            StorageError::InvalidData("No current version".to_string())
        })?;

        if &current == version {
            return Ok(()); // Already at target version
        }

        // Get all undo records from current to target
        let records = self.get_undo_records_for_version(version)?;

        let db = self.store.db.write();
        let cf_main = db
            .cf_handle(CF_MAIN)
            .ok_or_else(|| StorageError::DatabaseError("CF main not found".to_string()))?;
        let cf_meta = db
            .cf_handle(CF_META)
            .ok_or_else(|| StorageError::DatabaseError("CF meta not found".to_string()))?;

        let mut wb = WriteBatch::default();

        // Apply undo operations in reverse
        for record in records.iter().rev() {
            match &record.old_value {
                Some(value) => {
                    wb.put_cf(cf_main, &record.key, value);
                }
                None => {
                    wb.delete_cf(cf_main, &record.key);
                }
            }
        }

        // Update current version
        wb.put_cf(cf_meta, KEY_CURRENT_VERSION, version);

        db.write(wb)?;

        Ok(())
    }

    fn current_version(&self) -> Option<ADDigest> {
        let db = self.store.db.read();
        let cf = db.cf_handle(CF_META)?;
        db.get_cf(cf, KEY_CURRENT_VERSION)
            .ok()?
            .map(|v| Bytes::from(v))
    }

    fn available_versions(&self) -> Vec<ADDigest> {
        self.version_history.iter().cloned().collect()
    }

    fn prune_old_versions(&mut self) -> Result<()> {
        if self.version_history.len() <= self.keep_versions {
            return Ok(());
        }

        let to_remove = self.version_history.len() - self.keep_versions;
        let removed_versions: Vec<_> = self.version_history.drain(..to_remove).collect();

        let db = self.store.db.write();
        let cf_meta = db
            .cf_handle(CF_META)
            .ok_or_else(|| StorageError::DatabaseError("CF meta not found".to_string()))?;
        let cf_undo = db
            .cf_handle(CF_UNDO)
            .ok_or_else(|| StorageError::DatabaseError("CF undo not found".to_string()))?;

        let mut wb = WriteBatch::default();

        for version in removed_versions {
            // Remove version metadata
            wb.delete_cf(
                cf_meta,
                format!("version_{}", base16::encode_lower(&version)).as_bytes(),
            );

            // Remove associated undo records
            let records = self.get_undo_records_for_version(&version)?;
            for record in records {
                let inverted_lsn = !record.lsn;
                wb.delete_cf(cf_undo, inverted_lsn.to_be_bytes());
            }
        }

        db.write(wb)?;
        Ok(())
    }
}
