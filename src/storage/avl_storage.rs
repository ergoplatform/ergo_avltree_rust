use crate::batch_avl_prover::BatchAVLProver;
use crate::batch_node::*;
use crate::operation::{ADDigest, ADKey, ADValue};
use crate::storage::error::StorageError;
use crate::storage::kv_store::VersionedKVStore;
use crate::storage::rocksdb_store::VersionedRocksDBStore;
use crate::versioned_avl_storage::VersionedAVLStorage;
use alloc::boxed::Box;
use alloc::vec::Vec;
use anyhow::{anyhow, Result};
use bytes::Bytes;
use std::path::Path;
use std::sync::{Arc, Mutex};

const KEY_ROOT_HASH: &[u8] = b"__avl_root_hash__";
const KEY_ROOT_HEIGHT: &[u8] = b"__avl_root_height__";

/// Disk-backed AVL storage using RocksDB
pub struct DiskBackedAVLStorage {
    store: Arc<Mutex<VersionedRocksDBStore>>,
    key_length: usize,
    value_length: Option<usize>,
}

impl DiskBackedAVLStorage {
    pub fn new<P: AsRef<Path>>(
        path: P,
        key_length: usize,
        value_length: Option<usize>,
        keep_versions: usize,
    ) -> Result<Self> {
        let store = VersionedRocksDBStore::new(path, keep_versions)
            .map_err(|e| anyhow!("Failed to create storage: {}", e))?;

        Ok(Self {
            store: Arc::new(Mutex::new(store)),
            key_length,
            value_length,
        })
    }

    /// Create a resolver function that loads nodes from disk
    pub fn create_resolver(&self) -> Resolver {
        let store = Arc::clone(&self.store);
        let key_length = self.key_length;
        let value_length = self.value_length;

        move |label: &Digest32| -> Node {
            let store_guard = store.lock().unwrap();
            match store_guard.get(label) {
                Ok(Some(bytes)) => {
                    // Deserialize node from storage
                    let tree = AVLTree::new(dummy_resolver, key_length, value_length);
                    let node = tree.unpack(&Bytes::from(bytes));
                    node.borrow().clone()
                }
                _ => {
                    // If node not found, return label-only node
                    // This shouldn't happen in normal operation
                    Node::LabelOnly(NodeHeader::new(Some(*label), None))
                }
            }
        }
    }

    fn serialize_visited_nodes(&self, node: &NodeId, tree: &AVLTree) -> Vec<(Vec<u8>, Vec<u8>)> {
        let mut result = Vec::new();
        self.serialize_node_recursive(node, tree, &mut result);
        result
    }

    fn serialize_node_recursive(
        &self,
        node: &NodeId,
        tree: &AVLTree,
        result: &mut Vec<(Vec<u8>, Vec<u8>)>,
    ) {
        let node_ref = node.borrow();

        if !node_ref.visited() {
            return; // Only serialize visited nodes
        }

        match &*node_ref {
            Node::Internal(internal) => {
                // Serialize internal node
                let label = node_ref.get_label();
                let serialized = tree.pack(node.clone());
                result.push((label.to_vec(), serialized.to_vec()));

                // Recursively serialize children
                drop(node_ref);
                self.serialize_node_recursive(&internal.left, tree, result);
                self.serialize_node_recursive(&internal.right, tree, result);
            }
            Node::Leaf(_) => {
                // Serialize leaf node
                let label = node_ref.get_label();
                let serialized = tree.pack(node.clone());
                result.push((label.to_vec(), serialized.to_vec()));
            }
            Node::LabelOnly(_) => {
                // Label-only nodes don't need serialization
            }
        }
    }

    fn get_removed_node_labels(&self, prover: &mut BatchAVLProver) -> Vec<Vec<u8>> {
        let removed = prover.removed_nodes();
        removed
            .iter()
            .map(|node| node.borrow_mut().label().to_vec())
            .collect()
    }
}

impl VersionedAVLStorage for DiskBackedAVLStorage {
    fn update(
        &mut self,
        prover: &mut BatchAVLProver,
        additional_data: Vec<(ADKey, ADValue)>,
    ) -> Result<()> {
        let digest = prover
            .digest()
            .ok_or_else(|| anyhow!("No digest available"))?;

        // Serialize all visited (modified) nodes
        let mut to_insert = self.serialize_visited_nodes(&prover.top_node(), &prover.base.tree);

        // Add root metadata
        to_insert.push((KEY_ROOT_HASH.to_vec(), prover.top_node().borrow().get_label().to_vec()));
        to_insert.push((
            KEY_ROOT_HEIGHT.to_vec(),
            prover.base.tree.height.to_be_bytes().to_vec(),
        ));

        // Add additional data
        for (key, value) in additional_data {
            to_insert.push((key.to_vec(), value.to_vec()));
        }

        // Get labels of removed nodes
        let to_remove = self.get_removed_node_labels(prover);

        // Update storage with version
        let mut store = self.store.lock().unwrap();
        store
            .update_version(&digest, to_insert, to_remove)
            .map_err(|e| anyhow!("Storage update failed: {}", e))?;

        Ok(())
    }

    fn rollback(&mut self, version: &ADDigest) -> Result<(NodeId, usize)> {
        let mut store = self.store.lock().unwrap();

        // Rollback storage to target version
        store
            .rollback_to_version(version)
            .map_err(|e| anyhow!("Rollback failed: {}", e))?;

        // Load root node metadata
        let root_hash = store
            .get(KEY_ROOT_HASH)
            .map_err(|e| anyhow!("Failed to get root hash: {}", e))?
            .ok_or_else(|| anyhow!("Root hash not found"))?;

        let height_bytes = store
            .get(KEY_ROOT_HEIGHT)
            .map_err(|e| anyhow!("Failed to get root height: {}", e))?
            .ok_or_else(|| anyhow!("Root height not found"))?;

        let height = usize::from_be_bytes(
            height_bytes
                .try_into()
                .map_err(|_| anyhow!("Invalid height data"))?,
        );

        // Load root node from storage
        let mut digest: Digest32 = [0u8; 32];
        digest.copy_from_slice(&root_hash);

        // Create label-only node as root - it will be lazy-loaded when accessed
        let root_node = Node::new_label(&digest);

        Ok((root_node, height))
    }

    fn version(&self) -> Option<ADDigest> {
        let store = self.store.lock().unwrap();
        store.current_version()
    }

    fn rollback_versions<'a>(&'a self) -> Box<dyn Iterator<Item = ADDigest> + 'a> {
        let store = self.store.lock().unwrap();
        let versions = store.available_versions();
        Box::new(versions.into_iter())
    }
}

// Dummy resolver for deserialization (actual resolution happens via the storage resolver)
fn dummy_resolver(_label: &Digest32) -> Node {
    Node::LabelOnly(NodeHeader::new(Some(*_label), None))
}
