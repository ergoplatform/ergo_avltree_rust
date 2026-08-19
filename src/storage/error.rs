use std::fmt;

#[derive(Debug)]
pub enum StorageError {
    NotFound(String),
    VersionNotFound(String),
    DatabaseError(String),
    SerializationError(String),
    IOError(String),
    InvalidData(String),
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StorageError::NotFound(msg) => write!(f, "Not found: {}", msg),
            StorageError::VersionNotFound(msg) => write!(f, "Version not found: {}", msg),
            StorageError::DatabaseError(msg) => write!(f, "Database error: {}", msg),
            StorageError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            StorageError::IOError(msg) => write!(f, "I/O error: {}", msg),
            StorageError::InvalidData(msg) => write!(f, "Invalid data: {}", msg),
        }
    }
}

impl std::error::Error for StorageError {}

impl From<rocksdb::Error> for StorageError {
    fn from(err: rocksdb::Error) -> Self {
        StorageError::DatabaseError(err.to_string())
    }
}

impl From<bincode::Error> for StorageError {
    fn from(err: bincode::Error) -> Self {
        StorageError::SerializationError(err.to_string())
    }
}

impl From<std::io::Error> for StorageError {
    fn from(err: std::io::Error) -> Self {
        StorageError::IOError(err.to_string())
    }
}
