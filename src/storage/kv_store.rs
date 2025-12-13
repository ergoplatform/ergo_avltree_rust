use crate::operation::{ADDigest, ADKey, ADValue};
use crate::storage::error::StorageError;
use std::collections::HashMap;

pub type Result<T> = std::result::Result<T, StorageError>;

/// Batch of storage operations for atomic writes
#[derive(Default)]
pub struct StorageBatch {
    pub puts: HashMap<Vec<u8>, Vec<u8>>,
    pub deletes: Vec<Vec<u8>>,
}

impl StorageBatch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn put(&mut self, key: Vec<u8>, value: Vec<u8>) {
        self.puts.insert(key, value);
    }

    pub fn delete(&mut self, key: Vec<u8>) {
        self.deletes.push(key);
    }

    pub fn is_empty(&self) -> bool {
        self.puts.is_empty() && self.deletes.is_empty()
    }
}

/// Basic key-value store trait
pub trait KVStore {
    /// Get value by key
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>>;

    /// Put key-value pair
    fn put(&mut self, key: Vec<u8>, value: Vec<u8>) -> Result<()>;

    /// Delete key
    fn delete(&mut self, key: &[u8]) -> Result<()>;

    /// Execute batch operations atomically
    fn write_batch(&mut self, batch: StorageBatch) -> Result<()>;

    /// Check if key exists
    fn contains(&self, key: &[u8]) -> Result<bool> {
        Ok(self.get(key)?.is_some())
    }
}

/// Versioned key-value store with rollback capability
pub trait VersionedKVStore: KVStore {
    /// Update store with new version, including undo information
    fn update_version(
        &mut self,
        version: &ADDigest,
        to_insert: Vec<(Vec<u8>, Vec<u8>)>,
        to_remove: Vec<Vec<u8>>,
    ) -> Result<()>;

    /// Rollback to a specific version
    fn rollback_to_version(&mut self, version: &ADDigest) -> Result<()>;

    /// Get current version
    fn current_version(&self) -> Option<ADDigest>;

    /// Get list of available versions for rollback
    fn available_versions(&self) -> Vec<ADDigest>;

    /// Remove old versions beyond keep_versions limit
    fn prune_old_versions(&mut self) -> Result<()>;
}
