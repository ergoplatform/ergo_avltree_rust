//! AVL tree storage backed by `redb`.
//!
//! Implements the `VersionedAVLStorage` trait using `RedbVersionedStore` for
//! persistent, versioned AVL tree state with rollback support.

use crate::authenticated_tree_ops::AuthenticatedTreeOps;
use crate::batch_avl_prover::BatchAVLProver;
use crate::batch_node::*;
use crate::operation::*;
use crate::persistence::versioned_store::RedbVersionedStore;
use crate::versioned_avl_storage::VersionedAVLStorage;
use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use anyhow::{anyhow, Result};
use bytes::Bytes;
use std::path::{Path, PathBuf};

// Special keys in the node store for tree metadata
const TOP_NODE_HASH_KEY: &[u8] = b"__top_node_hash__";
const TOP_NODE_HEIGHT_KEY: &[u8] = b"__top_node_height__";

/// Persistent AVL storage backed by `redb`.
///
/// Mirrors the Scala `VersionedLDBAVLStorage`. Uses the existing `AVLTree::pack()`
/// and `AVLTree::unpack()` methods for node serialization, and `RedbVersionedStore`
/// for versioned persistence with rollback.
pub struct RedbAVLStorage {
    store: RedbVersionedStore,
    db_path: PathBuf,
    key_length: usize,
    value_length: Option<usize>,
}

impl RedbAVLStorage {
    /// Create a new persistent AVL storage at the given path.
    ///
    /// - `key_length`: fixed key length for the AVL tree
    /// - `value_length`: optional fixed value length (None for variable-length values)
    pub fn open(
        path: &Path,
        key_length: usize,
        value_length: Option<usize>,
    ) -> Result<Self> {
        let store = RedbVersionedStore::open(path)?;
        Ok(RedbAVLStorage {
            store,
            db_path: path.to_path_buf(),
            key_length,
            value_length,
        })
    }

    /// Create a `Resolver` closure that loads nodes from this storage.
    ///
    /// The resolver panics if a requested hash is not found — this indicates
    /// database corruption (the hash was referenced but the node data is missing).
    pub fn create_resolver(&self) -> Resolver {
        let path = self.db_path.clone();
        let key_length = self.key_length;
        let value_length = self.value_length;

        Arc::new(move |digest: &Digest32| {
            // Open a read-only handle to the DB.
            // redb supports concurrent readers safely.
            let db = redb::Database::open(&path)
                .expect("Failed to open DB in resolver");
            let txn = db.begin_read()
                .expect("Failed to begin read txn in resolver");
            let table = txn
                .open_table(redb::TableDefinition::<&[u8], &[u8]>::new("nodes"))
                .expect("Failed to open nodes table in resolver");

            match table.get(digest.as_slice()) {
                Ok(Some(guard)) => {
                    let bytes = Bytes::copy_from_slice(guard.value());
                    drop(guard);
                    drop(table);
                    drop(txn);

                    // Create a temporary AVLTree just for deserialization
                    let dummy_resolver: Resolver = Arc::new(|d: &Digest32| {
                        Node::LabelOnly(NodeHeader::new(Some(*d), None))
                    });
                    let dummy_tree = AVLTree::new(dummy_resolver, key_length, value_length);
                    let node_id = dummy_tree.unpack(&bytes);
                    let node = node_id.borrow().clone();
                    node
                }
                Ok(None) => {
                    panic!(
                        "Node not found in DB for digest {:02x}{:02x}{:02x}{:02x}... — database is corrupted",
                        digest[0], digest[1], digest[2], digest[3]
                    );
                }
                Err(e) => {
                    panic!("DB read error in resolver: {}", e);
                }
            }
        })
    }

    /// Collect all nodes from the tree by walking it recursively.
    /// Returns (label, packed_bytes) pairs for all nodes in the tree.
    fn collect_all_nodes(
        tree: &AVLTree,
        node: &NodeId,
        result: &mut Vec<(Vec<u8>, Vec<u8>)>,
    ) {
        let n = node.borrow().clone();
        match &n {
            Node::LabelOnly(_) => {
                // Not resolved / not dirty, skip
            }
            Node::Internal(internal) => {
                let label = node.borrow_mut().label();
                let packed = tree.pack(node.clone());
                result.push((label.to_vec(), packed.to_vec()));
                Self::collect_all_nodes(tree, &internal.left, result);
                Self::collect_all_nodes(tree, &internal.right, result);
            }
            Node::Leaf(_) => {
                let label = node.borrow_mut().label();
                let packed = tree.pack(node.clone());
                result.push((label.to_vec(), packed.to_vec()));
            }
        }
    }
}

impl VersionedAVLStorage for RedbAVLStorage {
    fn update(
        &mut self,
        prover: &mut BatchAVLProver,
        _additional_data: Vec<(ADKey, ADValue)>,
    ) -> Result<()> {
        let new_digest = prover
            .digest()
            .ok_or_else(|| anyhow!("Prover has no digest"))?;

        let tree = prover.get_tree();
        let root = prover.top_node();

        // Collect all nodes to persist
        let mut node_pairs = Vec::new();
        Self::collect_all_nodes(tree, &root, &mut node_pairs);

        // Build metadata entries
        let height = tree.height;
        let root_label = root.borrow_mut().label();
        let height_bytes = (height as u64).to_le_bytes();

        // Build insert list: all nodes + metadata
        let mut to_insert: Vec<(&[u8], &[u8])> = node_pairs
            .iter()
            .map(|(k, v)| (k.as_slice(), v.as_slice()))
            .collect();
        let root_label_vec = root_label.to_vec();
        to_insert.push((TOP_NODE_HASH_KEY, root_label_vec.as_slice()));
        to_insert.push((TOP_NODE_HEIGHT_KEY, &height_bytes));

        // Collect removed node labels
        let removed = prover.removed_nodes();
        let removed_labels: Vec<Vec<u8>> = removed
            .iter()
            .map(|n| n.borrow().get_label().to_vec())
            .collect();
        let to_remove: Vec<&[u8]> = removed_labels.iter().map(|l| l.as_slice()).collect();

        self.store.update(&new_digest, &to_insert, &to_remove)?;
        Ok(())
    }

    fn rollback(&mut self, version: &ADDigest) -> Result<(NodeId, usize)> {
        self.store.rollback(version)?;

        // Read root hash and height from the store
        let root_hash = self
            .store
            .get_node(TOP_NODE_HASH_KEY)?
            .ok_or_else(|| anyhow!("Root hash not found after rollback"))?;
        let height_bytes = self
            .store
            .get_node(TOP_NODE_HEIGHT_KEY)?
            .ok_or_else(|| anyhow!("Height not found after rollback"))?;
        let height = u64::from_le_bytes(height_bytes.as_slice().try_into()?) as usize;

        // Read the root node from the store
        let root_data = self
            .store
            .get_node(&root_hash)?
            .ok_or_else(|| anyhow!("Root node data not found after rollback"))?;

        let root_bytes = Bytes::from(root_data);
        let tree = AVLTree::new(
            self.create_resolver(),
            self.key_length,
            self.value_length,
        );
        let root_node = tree.unpack(&root_bytes);

        Ok((root_node, height))
    }

    fn version(&self) -> Option<ADDigest> {
        self.store
            .last_version_id()
            .ok()
            .flatten()
            .map(Bytes::from)
    }

    fn rollback_versions<'a>(&'a self) -> Box<dyn Iterator<Item = ADDigest> + 'a> {
        let versions = self.store.rollback_versions().unwrap_or_default();
        Box::new(versions.into_iter().map(Bytes::from))
    }
}
