#[cfg(feature = "disk-persistence")]
pub mod error;
#[cfg(feature = "disk-persistence")]
pub mod kv_store;
#[cfg(feature = "disk-persistence")]
pub mod rocksdb_store;
#[cfg(feature = "disk-persistence")]
pub mod avl_storage;

#[cfg(feature = "disk-persistence")]
pub use error::StorageError;
#[cfg(feature = "disk-persistence")]
pub use kv_store::{KVStore, VersionedKVStore, StorageBatch};
#[cfg(feature = "disk-persistence")]
pub use rocksdb_store::{RocksDBStore, VersionedRocksDBStore};
#[cfg(feature = "disk-persistence")]
pub use avl_storage::DiskBackedAVLStorage;
