use crate::authenticated_tree_ops::AuthenticatedTreeOps;
use crate::batch_avl_prover::BatchAVLProver;
use crate::batch_node::*;
use crate::operation::{ADDigest, ADKey, ADValue, Digest32};
use crate::storage::kv_store::{KVStore, VersionedKVStore};
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
}

// Global storage context for disk-backed resolver
struct GlobalStorageContext {
    store: Arc<Mutex<VersionedRocksDBStore>>,
    key_length: usize,
    value_length: Option<usize>,
}

static mut GLOBAL_STORAGE: Option<GlobalStorageContext> = None;

fn disk_backed_resolver(label: &Digest32) -> Node {
    unsafe {
        if let Some(context) = &*(&raw const GLOBAL_STORAGE) {
            let store_guard = context.store.lock().unwrap();
            match store_guard.get(label) {
                Ok(Some(bytes)) => {
                    // Create a temporary tree for deserialization using the correct config
                    let temp_tree = AVLTree::new(
                        dummy_resolver, 
                        context.key_length, 
                        context.value_length
                    );
                    
                    // Try to deserialize
                    // We need to catch panics because unpack might panic on invalid data
                    // But unpack doesn't return Result, it panics. 
                    // For now, assume data is correct if it exists.
                    let node = temp_tree.unpack(&Bytes::from(bytes));
                    
                    // Clone the node contents to return an owned Node
                    let cloned = node.borrow().clone();
                    cloned
                }
                Ok(None) => {
                    println!("Node not found in storage: {:?}", label);
                    Node::LabelOnly(NodeHeader::new(Some(*label), None))
                }
                Err(e) => {
                    println!("Storage error for node {:?}: {}", label, e);
                    Node::LabelOnly(NodeHeader::new(Some(*label), None))
                }
            }
        } else {
            Node::LabelOnly(NodeHeader::new(Some(*label), None))
        }
    }
}

impl Drop for DiskBackedAVLStorage {
    fn drop(&mut self) {
        unsafe {
            // Clear global storage when the storage instance is dropped
            // This releases the RocksDB lock
            GLOBAL_STORAGE = None;
        }
    }
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
        
        let store_arc = Arc::new(Mutex::new(store));
        
        // Set global storage for resolver
        unsafe {
            GLOBAL_STORAGE = Some(GlobalStorageContext {
                store: Arc::clone(&store_arc),
                key_length,
                value_length,
            });
        }

        Ok(Self {
            store: store_arc,
        })
    }

    /// Get the resolver function for this storage
    pub fn get_resolver() -> Resolver {
        disk_backed_resolver
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
        if matches!(&*node.borrow(), Node::LabelOnly(_)) {
            return;
        }

        let is_internal = matches!(&*node.borrow(), Node::Internal(_));
        let label = node.borrow().get_label();
        
        if is_internal {
            let (left, right) = {
                if let Node::Internal(internal) = &*node.borrow() {
                    (internal.left.clone(), internal.right.clone())
                } else {
                    return;
                }
            };
            
            // Serialize internal node
            let serialized = tree.pack(node.clone());
            result.push((label.to_vec(), serialized.to_vec()));

            // Recursively serialize children
            self.serialize_node_recursive(&left, tree, result);
            self.serialize_node_recursive(&right, tree, result);
        } else {
            // Serialize leaf node
            let serialized = tree.pack(node.clone());
            result.push((label.to_vec(), serialized.to_vec()));
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
        let top_node = prover.top_node();
        let mut to_insert = self.serialize_visited_nodes(&top_node, &prover.base.tree);

        // Add root metadata
        to_insert.push((KEY_ROOT_HASH.to_vec(), top_node.borrow().get_label().to_vec()));
        to_insert.push((
            KEY_ROOT_HEIGHT.to_vec(),
            prover.base.tree.height.to_be_bytes().to_vec(),
        ));

        // Add additional data
        for (key, value) in additional_data {
            to_insert.push((key.to_vec(), value.to_vec()));
        }

        // Get labels of removed nodes
        let mut to_remove = self.get_removed_node_labels(prover);
        
        // Filter out nodes that are being inserted (they are still in the tree)
        let inserted_keys: std::collections::HashSet<_> = to_insert.iter().map(|(k, _)| k.clone()).collect();
        to_remove.retain(|k| !inserted_keys.contains(k));

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
        
        println!("Rollback complete. New root hash: {:?}", root_hash);

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
