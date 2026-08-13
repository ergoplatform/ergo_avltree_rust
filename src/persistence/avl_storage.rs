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
use anyhow::{anyhow, ensure, Result};
use bytes::Bytes;
use std::path::Path;

// Special keys in the node store for tree metadata
const TOP_NODE_HASH_KEY: &[u8] = b"__top_node_hash__";
const TOP_NODE_HEIGHT_KEY: &[u8] = b"__top_node_height__";

/// Persistent AVL storage backed by `redb`.
///
/// Mirrors the Scala `VersionedLDBAVLStorage`. Uses the existing `AVLTree::pack()`
/// and `AVLTree::unpack()` methods for node serialization, and `RedbVersionedStore`
/// for versioned persistence with rollback.
pub struct RedbAVLStorage {
    store: Arc<RedbVersionedStore>,
    key_length: usize,
    value_length: Option<usize>,
}

impl RedbAVLStorage {
    fn cached_label(node: &Node) -> Option<Digest32> {
        match node {
            Node::LabelOnly(header) => header.label,
            Node::Internal(internal) => internal.hdr.label,
            Node::Leaf(leaf) => leaf.hdr.label,
        }
    }

    fn unpack_verified_node(
        tree: &AVLTree,
        bytes: &Bytes,
        expected_digest: &Digest32,
    ) -> Result<NodeId> {
        let node = tree.unpack(bytes);
        let actual_digest = node.borrow_mut().label();
        ensure!(
            actual_digest == *expected_digest,
            "Persisted node digest does not match its content-address key"
        );
        Ok(node)
    }

    /// Create a new persistent AVL storage at the given path.
    pub fn open(path: &Path, key_length: usize, value_length: Option<usize>) -> Result<Self> {
        let store = RedbVersionedStore::open(path)?;
        Ok(RedbAVLStorage {
            store: Arc::new(store),
            key_length,
            value_length,
        })
    }

    /// Create a `Resolver` closure that loads nodes from this storage.
    ///
    /// Uses a shared `Arc<RedbVersionedStore>` — no file re-opening per call.
    /// Panics if a requested hash is not found (= database corruption).
    pub fn create_resolver(&self) -> Resolver {
        let store = Arc::clone(&self.store);
        let key_length = self.key_length;
        let value_length = self.value_length;

        Arc::new(move |digest: &Digest32| {
            match store.get_node(digest.as_slice()) {
                Ok(Some(data)) => {
                    let bytes = Bytes::from(data);
                    // Temporary AVLTree for deserialization only
                    let dummy_resolver: Resolver =
                        Arc::new(|d: &Digest32| Node::LabelOnly(NodeHeader::new(Some(*d), None)));
                    let dummy_tree =
                        AVLTree::with_resolver(dummy_resolver, key_length, value_length);
                    let node_id = Self::unpack_verified_node(&dummy_tree, &bytes, digest)
                        .expect("Persisted node failed content-address verification");
                    let node = node_id.borrow().clone();
                    node
                }
                Ok(None) => {
                    panic!(
                        "Node not found in DB for digest {:02x}{:02x}{:02x}{:02x}... -- database is corrupted",
                        digest[0], digest[1], digest[2], digest[3]
                    );
                }
                Err(e) => {
                    panic!("DB read error in resolver: {}", e);
                }
            }
        })
    }

    /// Collect all resolved nodes from the tree by walking it recursively.
    /// Returns (label, packed_bytes) pairs for visited (non-LabelOnly) nodes.
    fn collect_all_nodes(tree: &AVLTree, node: &NodeId, result: &mut Vec<(Vec<u8>, Vec<u8>)>) {
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

        // Collect all resolved nodes
        let mut node_pairs = Vec::new();
        Self::collect_all_nodes(tree, &root, &mut node_pairs);

        // Build metadata entries
        let height = tree.height;
        let root_label = root.borrow_mut().label();
        let height_bytes = (height as u64).to_le_bytes();
        let root_label_vec = root_label.to_vec();

        if new_digest.len() != 33
            || new_digest[..32] != root_label_vec
            || new_digest[32] as usize != height
        {
            return Err(anyhow!(
                "Prover digest does not match its AVL root metadata"
            ));
        }

        // Collect removed node labels before deciding whether this update is a
        // verified proof-only no-op or a new persisted version.
        let removed = prover.removed_nodes();
        let removed_labels: Vec<Vec<u8>> = removed
            .iter()
            .map(|node| {
                Self::cached_label(&node.borrow())
                    .map(|label| label.to_vec())
                    .ok_or_else(|| {
                        anyhow!(
                            "Removed node has no cached persistence identity; persist the baseline before applying operations"
                        )
                    })
            })
            .collect::<Result<_>>()?;
        let to_remove: Vec<&[u8]> = removed_labels.iter().map(|l| l.as_slice()).collect();

        // A proof-only batch such as Lookup can leave the AVL digest unchanged.
        // Coalesce it only after verifying every materialized node, root
        // metadata entry, and removal candidate against the persisted tip.
        if self.store.last_version_id()?.as_deref() == Some(new_digest.as_ref()) {
            let mut expected_entries: Vec<(&[u8], &[u8])> = node_pairs
                .iter()
                .map(|(key, value)| (key.as_slice(), value.as_slice()))
                .collect();
            expected_entries.push((TOP_NODE_HASH_KEY, root_label_vec.as_slice()));
            expected_entries.push((TOP_NODE_HEIGHT_KEY, &height_bytes));
            self.store
                .verify_current_state(&new_digest, &expected_entries, &to_remove)?;
            return Ok(());
        }

        // Q2 FIX: Filter out content-addressed nodes that already exist in DB.
        // Since keys are Blake2b hashes, if the key exists, the content is
        // guaranteed identical (content-addressed = idempotent). This eliminates
        // 100% of write amplification from re-persisting unchanged nodes.
        let mut to_insert: Vec<(&[u8], &[u8])> = Vec::new();
        for (key, value) in &node_pairs {
            if self.store.get_node(key)?.is_none() {
                to_insert.push((key.as_slice(), value.as_slice()));
            }
        }
        // Metadata keys ALWAYS update (not content-addressed)
        to_insert.push((TOP_NODE_HASH_KEY, root_label_vec.as_slice()));
        to_insert.push((TOP_NODE_HEIGHT_KEY, &height_bytes));

        self.store.update(&new_digest, &to_insert, &to_remove)?;
        Ok(())
    }

    fn rollback(&mut self, version: &ADDigest) -> Result<(NodeId, usize)> {
        self.store.rollback(version)?;

        let root_hash = self
            .store
            .get_node(TOP_NODE_HASH_KEY)?
            .ok_or_else(|| anyhow!("Root hash not found after rollback"))?;
        let height_bytes = self
            .store
            .get_node(TOP_NODE_HEIGHT_KEY)?
            .ok_or_else(|| anyhow!("Height not found after rollback"))?;
        let height = u64::from_le_bytes(height_bytes.as_slice().try_into()?) as usize;

        let root_data = self
            .store
            .get_node(&root_hash)?
            .ok_or_else(|| anyhow!("Root node data not found after rollback"))?;

        let root_bytes = Bytes::from(root_data);
        let root_digest: Digest32 = root_hash
            .as_slice()
            .try_into()
            .map_err(|_| anyhow!("Invalid persisted root hash length"))?;
        let tree =
            AVLTree::with_resolver(self.create_resolver(), self.key_length, self.value_length);
        let root_node = Self::unpack_verified_node(&tree, &root_bytes, &root_digest)?;

        Ok((root_node, height))
    }

    fn version(&self) -> Option<ADDigest> {
        self.store.last_version_id().ok().flatten().map(Bytes::from)
    }

    fn rollback_versions<'a>(&'a self) -> Box<dyn Iterator<Item = ADDigest> + 'a> {
        let versions = self.store.rollback_versions().unwrap_or_default();
        Box::new(versions.into_iter().map(Bytes::from))
    }

    fn flush(&self) -> Result<()> {
        self.store.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;
    use tempfile::tempdir;

    fn dummy_resolver(digest: &Digest32) -> Node {
        Node::LabelOnly(NodeHeader::new(Some(*digest), None))
    }

    fn packed_leaf() -> (AVLTree, Bytes, Digest32) {
        let tree = AVLTree::new(dummy_resolver, 32, None);
        let key = Bytes::from(vec![1u8; 32]);
        let value = Bytes::from("value");
        let next_key = Bytes::from(vec![2u8; 32]);
        let leaf = LeafNode::new(&key, &value, &next_key);
        let digest = leaf.borrow_mut().label();
        let bytes = tree.pack(leaf);
        (tree, bytes, digest)
    }

    #[test]
    fn fresh_unlabelled_node_has_no_persisted_removal_key() {
        let key = Bytes::from(vec![1u8; 32]);
        let value = Bytes::from("value");
        let next_key = Bytes::from(vec![2u8; 32]);
        let leaf = LeafNode::new(&key, &value, &next_key);

        assert_eq!(RedbAVLStorage::cached_label(&leaf.borrow()), None);
    }

    #[test]
    fn raw_mutation_without_a_persisted_baseline_is_rejected() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("avl.redb");
        let mut storage = RedbAVLStorage::open(&path, 32, None).unwrap();
        let tree = AVLTree::new(dummy_resolver, 32, None);
        let mut prover = BatchAVLProver::new(tree, true);
        let key = Bytes::from(vec![1u8; 32]);
        let value = Bytes::from("value");
        prover
            .perform_one_operation(&Operation::Insert(KeyValue { key, value }))
            .unwrap();

        let error = storage.update(&mut prover, vec![]).unwrap_err();

        assert_eq!(
            error.to_string(),
            "Removed node has no cached persistence identity; persist the baseline before applying operations"
        );
        assert_eq!(storage.version(), None);
    }

    #[test]
    fn verified_unpack_caches_the_content_address() {
        let (tree, bytes, digest) = packed_leaf();

        let node = RedbAVLStorage::unpack_verified_node(&tree, &bytes, &digest).unwrap();

        assert_eq!(RedbAVLStorage::cached_label(&node.borrow()), Some(digest));
    }

    #[test]
    fn verified_unpack_rejects_a_mismatched_content_address() {
        let (tree, bytes, mut digest) = packed_leaf();
        digest[0] ^= 1;

        let result = RedbAVLStorage::unpack_verified_node(&tree, &bytes, &digest);

        assert!(result.is_err());
    }
}
