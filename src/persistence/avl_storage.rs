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
use alloc::collections::BTreeSet;
use alloc::sync::Arc;
use alloc::vec::Vec;
use anyhow::{anyhow, ensure, Result};
use bytes::Bytes;
use std::path::Path;

// Special keys in the node store for tree metadata
const TOP_NODE_HASH_KEY: &[u8] = b"__top_node_hash__";
const TOP_NODE_HEIGHT_KEY: &[u8] = b"__top_node_height__";

type SnapshotLookup<'a> = dyn Fn(&[u8]) -> Result<Option<Vec<u8>>> + 'a;

struct ValidatedSubtree {
    min_key: ADKey,
    max_key: ADKey,
    last_next_key: ADKey,
    height: usize,
}

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
        Self::validate_packed_node(bytes, tree.key_length, tree.value_length)?;
        let node = tree.unpack(bytes);
        let actual_digest = node.borrow_mut().label();
        ensure!(
            actual_digest == *expected_digest,
            "Persisted node digest does not match its content-address key"
        );
        Ok(node)
    }

    fn validate_packed_node(
        bytes: &[u8],
        key_length: usize,
        value_length: Option<usize>,
    ) -> Result<()> {
        let prefix = *bytes
            .first()
            .ok_or_else(|| anyhow!("Persisted node is empty"))?;
        match prefix {
            0 => {
                let expected = 2usize
                    .checked_add(key_length)
                    .and_then(|length| length.checked_add(64))
                    .ok_or_else(|| anyhow!("Persisted internal-node length overflow"))?;
                ensure!(
                    bytes.len() == expected,
                    "Persisted internal node has a noncanonical length"
                );
                ensure!(
                    matches!(bytes[1] as i8, -1..=1),
                    "Persisted internal node has an invalid balance"
                );
            }
            1 => {
                let prefix_and_key = 1usize
                    .checked_add(key_length)
                    .ok_or_else(|| anyhow!("Persisted leaf-node key length overflow"))?;
                let (encoded_length_size, value_length) = match value_length {
                    Some(length) => (0usize, length),
                    None => {
                        let length_end = prefix_and_key.checked_add(4).ok_or_else(|| {
                            anyhow!("Persisted leaf-node value-length offset overflow")
                        })?;
                        let length_bytes = bytes
                            .get(prefix_and_key..length_end)
                            .ok_or_else(|| anyhow!("Persisted leaf value length is truncated"))?;
                        let encoded = u32::from_be_bytes(length_bytes.try_into()?);
                        (
                            4,
                            usize::try_from(encoded).map_err(|_| {
                                anyhow!("Persisted leaf value length exceeds platform usize")
                            })?,
                        )
                    }
                };
                let expected = prefix_and_key
                    .checked_add(encoded_length_size)
                    .and_then(|length| length.checked_add(value_length))
                    .and_then(|length| length.checked_add(key_length))
                    .ok_or_else(|| anyhow!("Persisted leaf-node total length overflow"))?;
                ensure!(
                    bytes.len() == expected,
                    "Persisted leaf node has a noncanonical length"
                );
            }
            _ => return Err(anyhow!("Persisted node has an invalid prefix")),
        }
        Ok(())
    }

    fn validate_reachable_node(
        lookup: &SnapshotLookup<'_>,
        tree: &AVLTree,
        expected_digest: &Digest32,
        depth: usize,
        active_path: &mut BTreeSet<Digest32>,
        seen: &mut BTreeSet<Digest32>,
    ) -> Result<(NodeId, ValidatedSubtree)> {
        ensure!(
            depth <= u8::MAX as usize,
            "Persisted AVL depth exceeds digest encoding"
        );
        ensure!(
            !active_path.contains(expected_digest),
            "Cycle detected in reachable AVL nodes"
        );
        ensure!(
            seen.insert(*expected_digest),
            "Persisted AVL digest is reachable more than once"
        );
        ensure!(
            active_path.insert(*expected_digest),
            "Cycle detected in reachable AVL nodes"
        );

        let result = (|| -> Result<(NodeId, ValidatedSubtree)> {
            let data =
                lookup(expected_digest)?.ok_or_else(|| anyhow!("Reachable AVL node is missing"))?;
            let bytes = Bytes::from(data);
            let node = Self::unpack_verified_node(tree, &bytes, expected_digest)?;
            let node_value = node.borrow().clone();
            let summary = match node_value {
                Node::LabelOnly(_) => {
                    return Err(anyhow!("Persisted AVL record remained unresolved"));
                }
                Node::Leaf(leaf) => {
                    let key = leaf
                        .hdr
                        .key
                        .ok_or_else(|| anyhow!("Persisted leaf key is missing"))?;
                    ensure!(
                        key < leaf.next_node_key,
                        "Persisted AVL leaf key must precede its next key"
                    );
                    ValidatedSubtree {
                        min_key: key.clone(),
                        max_key: key,
                        last_next_key: leaf.next_node_key,
                        height: 0,
                    }
                }
                Node::Internal(internal) => {
                    let separator = internal
                        .hdr
                        .key
                        .ok_or_else(|| anyhow!("Persisted internal separator is missing"))?;
                    let left_digest = Self::cached_label(&internal.left.borrow())
                        .ok_or_else(|| anyhow!("Persisted left-child digest is missing"))?;
                    let right_digest = Self::cached_label(&internal.right.borrow())
                        .ok_or_else(|| anyhow!("Persisted right-child digest is missing"))?;
                    let (_, left) = Self::validate_reachable_node(
                        lookup,
                        tree,
                        &left_digest,
                        depth
                            .checked_add(1)
                            .ok_or_else(|| anyhow!("Persisted AVL depth overflow"))?,
                        active_path,
                        seen,
                    )?;
                    let (_, right) = Self::validate_reachable_node(
                        lookup,
                        tree,
                        &right_digest,
                        depth
                            .checked_add(1)
                            .ok_or_else(|| anyhow!("Persisted AVL depth overflow"))?,
                        active_path,
                        seen,
                    )?;

                    ensure!(
                        separator == right.min_key,
                        "Persisted AVL separator does not equal the minimum key of its right subtree"
                    );
                    ensure!(
                        left.last_next_key == right.min_key,
                        "Persisted AVL leaf next-key chain is not contiguous"
                    );
                    ensure!(
                        left.max_key < right.min_key,
                        "Persisted AVL subtree keys are not strictly ordered"
                    );
                    let expected_balance = if right.height == left.height {
                        0
                    } else if right.height
                        == left
                            .height
                            .checked_add(1)
                            .ok_or_else(|| anyhow!("Persisted AVL subtree height overflow"))?
                    {
                        1
                    } else if left.height
                        == right
                            .height
                            .checked_add(1)
                            .ok_or_else(|| anyhow!("Persisted AVL subtree height overflow"))?
                    {
                        -1
                    } else {
                        return Err(anyhow!(
                            "Persisted AVL subtree heights violate the balance bound"
                        ));
                    };
                    ensure!(
                        internal.balance == expected_balance,
                        "Persisted AVL balance does not match subtree heights"
                    );
                    let height = core::cmp::max(left.height, right.height)
                        .checked_add(1)
                        .ok_or_else(|| anyhow!("Persisted AVL subtree height overflow"))?;
                    ensure!(
                        height <= u8::MAX as usize,
                        "Persisted AVL height exceeds digest encoding"
                    );
                    ValidatedSubtree {
                        min_key: left.min_key,
                        max_key: right.max_key,
                        last_next_key: right.last_next_key,
                        height,
                    }
                }
            };
            Ok((node, summary))
        })();

        active_path.remove(expected_digest);
        result
    }

    fn validate_and_load_snapshot(
        lookup: &SnapshotLookup<'_>,
        version: &[u8],
        key_length: usize,
        value_length: Option<usize>,
        validation_resolver: Resolver,
    ) -> Result<(NodeId, usize)> {
        ensure!(
            version.len() == 33,
            "Target AVL version must be exactly 33 bytes"
        );
        let root_hash = lookup(TOP_NODE_HASH_KEY)?
            .ok_or_else(|| anyhow!("Root hash not found in rollback target"))?;
        let root_digest: Digest32 = root_hash
            .as_slice()
            .try_into()
            .map_err(|_| anyhow!("Persisted root hash must be exactly 32 bytes"))?;

        let height_bytes = lookup(TOP_NODE_HEIGHT_KEY)?
            .ok_or_else(|| anyhow!("Height not found in rollback target"))?;
        let encoded_height: [u8; 8] = height_bytes
            .as_slice()
            .try_into()
            .map_err(|_| anyhow!("Persisted height must be exactly 8 bytes"))?;
        let height_u64 = u64::from_le_bytes(encoded_height);
        ensure!(
            height_u64 <= u8::MAX as u64,
            "Persisted AVL height exceeds digest encoding"
        );
        let height = usize::try_from(height_u64)
            .map_err(|_| anyhow!("Persisted AVL height exceeds platform usize"))?;
        ensure!(
            version[..32] == root_digest && version[32] == height_u64 as u8,
            "Target AVL version does not match persisted root metadata"
        );

        let tree = AVLTree::with_resolver(validation_resolver, key_length, value_length);
        let mut active_path = BTreeSet::new();
        let mut seen = BTreeSet::new();
        let (root, summary) = Self::validate_reachable_node(
            lookup,
            &tree,
            &root_digest,
            0,
            &mut active_path,
            &mut seen,
        )?;
        ensure!(
            summary.min_key == tree.negative_infinity_key(),
            "Persisted AVL tree is missing its negative-infinity sentinel"
        );
        ensure!(
            summary.last_next_key == tree.positive_infinity_key(),
            "Persisted AVL tree is missing its positive-infinity sentinel"
        );
        ensure!(
            summary.height == height,
            "Computed AVL height does not match persisted metadata"
        );
        Ok((root, height))
    }

    /// Create a new persistent AVL storage at the given path.
    pub fn open(path: &Path, key_length: usize, value_length: Option<usize>) -> Result<Self> {
        ensure!(key_length > 0, "AVL key length must be greater than zero");
        let store = RedbVersionedStore::open(path)?;
        if let Some(version) = store.last_version_id()? {
            store.rollback_with_validation(&version, |lookup| {
                Self::validate_and_load_snapshot(
                    lookup,
                    &version,
                    key_length,
                    value_length,
                    Arc::new(|digest| Node::LabelOnly(NodeHeader::new(Some(*digest), None))),
                )
            })?;
        }
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
        for (_, packed) in &node_pairs {
            Self::validate_packed_node(packed, tree.key_length, tree.value_length)?;
        }

        // Build metadata entries
        let height = tree.height;
        ensure!(
            height <= u8::MAX as usize,
            "AVL height exceeds digest encoding"
        );
        let root_label = root.borrow_mut().label();
        let height_u64 = u64::try_from(height)
            .map_err(|_| anyhow!("AVL height exceeds persistence encoding"))?;
        let height_bytes = height_u64.to_le_bytes();
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

        let content_addressed: Vec<(&[u8], &[u8])> = node_pairs
            .iter()
            .map(|(key, value)| (key.as_slice(), value.as_slice()))
            .collect();
        let metadata: [(&[u8], &[u8]); 2] = [
            (TOP_NODE_HASH_KEY, root_label_vec.as_slice()),
            (TOP_NODE_HEIGHT_KEY, &height_bytes),
        ];
        self.store
            .update_partitioned(&new_digest, &content_addressed, &metadata, &to_remove)?;
        Ok(())
    }

    fn rollback(&mut self, version: &ADDigest) -> Result<(NodeId, usize)> {
        let key_length = self.key_length;
        let value_length = self.value_length;
        self.store.rollback_with_validation(version, |lookup| {
            Self::validate_and_load_snapshot(
                lookup,
                version,
                key_length,
                value_length,
                Arc::new(|digest| Node::LabelOnly(NodeHeader::new(Some(*digest), None))),
            )
        })
    }

    fn version(&self) -> Option<ADDigest> {
        self.try_version()
            .expect("Redb current-version metadata is corrupt")
    }

    fn try_version(&self) -> Result<Option<ADDigest>> {
        Ok(self.store.last_version_id()?.map(Bytes::from))
    }

    fn rollback_versions<'a>(&'a self) -> Box<dyn Iterator<Item = ADDigest> + 'a> {
        self.try_rollback_versions()
            .expect("Redb version history is corrupt")
    }

    fn try_rollback_versions<'a>(&'a self) -> Result<Box<dyn Iterator<Item = ADDigest> + 'a>> {
        let versions = self.store.rollback_versions()?;
        Ok(Box::new(versions.into_iter().map(Bytes::from)))
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
    use std::panic::{catch_unwind, AssertUnwindSafe};
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

    fn persist_root(storage: &mut RedbAVLStorage, root: NodeId, height: usize) -> ADDigest {
        let mut tree = AVLTree::new(dummy_resolver, 32, None);
        tree.root = Some(root);
        tree.height = height;
        let mut prover = BatchAVLProver::new(tree, true);
        storage.update(&mut prover, vec![]).unwrap();
        prover.digest().unwrap()
    }

    fn repeated_key(byte: u8) -> Bytes {
        Bytes::from(vec![byte; 32])
    }

    fn two_leaf_root(
        separator: Bytes,
        left_next: Bytes,
        balance: Balance,
    ) -> (NodeId, Digest32, Digest32) {
        let left = LeafNode::new(&repeated_key(0), &Bytes::new(), &left_next);
        let right = LeafNode::new(&repeated_key(1), &Bytes::from("right"), &repeated_key(0xFF));
        let left_digest = left.borrow_mut().label();
        let right_digest = right.borrow_mut().label();
        (
            InternalNode::new(Some(separator), &left, &right, balance),
            left_digest,
            right_digest,
        )
    }

    fn assert_authentic_record(storage: &RedbAVLStorage, digest: &Digest32) {
        let data = storage.store.get_node(digest).unwrap().unwrap();
        let bytes = Bytes::from(data);
        let tree = AVLTree::new(dummy_resolver, 32, None);
        RedbAVLStorage::validate_packed_node(&bytes, 32, None).unwrap();
        RedbAVLStorage::unpack_verified_node(&tree, &bytes, digest).unwrap();
    }

    fn assert_target_rollback_error_is_atomic(
        storage: &mut RedbAVLStorage,
        target: &ADDigest,
        expected_error: &str,
    ) {
        let tip = [0xA5u8; 33];
        storage
            .store
            .update(&tip, &[(b"after-target", b"still-current")], &[])
            .unwrap();
        let root_before = storage.store.get_node(TOP_NODE_HASH_KEY).unwrap();
        let height_before = storage.store.get_node(TOP_NODE_HEIGHT_KEY).unwrap();
        let versions_before = storage.store.rollback_versions().unwrap();

        let error = storage.rollback(target).unwrap_err();

        assert!(
            error.to_string().contains(expected_error),
            "unexpected rollback error: {error:#}"
        );
        assert_eq!(storage.store.last_version_id().unwrap(), Some(tip.to_vec()));
        assert_eq!(storage.store.rollback_versions().unwrap(), versions_before);
        assert_eq!(
            storage.store.get_node(b"after-target").unwrap(),
            Some(b"still-current".to_vec())
        );
        assert_eq!(
            storage.store.get_node(TOP_NODE_HASH_KEY).unwrap(),
            root_before
        );
        assert_eq!(
            storage.store.get_node(TOP_NODE_HEIGHT_KEY).unwrap(),
            height_before
        );
    }

    #[test]
    fn rollback_rejects_a_shared_reachable_digest_before_a_second_lookup() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let dir = tempdir().unwrap();
        let mut storage = RedbAVLStorage::open(&dir.path().join("test.redb"), 32, None).unwrap();
        let shared = LeafNode::new(&repeated_key(0), &Bytes::new(), &repeated_key(0xFF));
        let shared_digest = shared.borrow_mut().label();
        let root = InternalNode::new(Some(repeated_key(0)), &shared, &shared, 0);
        let target = persist_root(&mut storage, root, 1);
        let shared_lookups = AtomicUsize::new(0);

        let error = storage
            .store
            .rollback_with_validation(&target, |lookup| {
                let counted_lookup = |key: &[u8]| {
                    if key == shared_digest.as_slice() {
                        shared_lookups.fetch_add(1, Ordering::SeqCst);
                    }
                    lookup(key)
                };
                RedbAVLStorage::validate_and_load_snapshot(
                    &counted_lookup,
                    &target,
                    32,
                    None,
                    Arc::new(|digest| Node::LabelOnly(NodeHeader::new(Some(*digest), None))),
                )
            })
            .unwrap_err();

        assert_eq!(
            shared_lookups.load(Ordering::SeqCst),
            1,
            "a shared digest must be rejected before a second database lookup"
        );
        assert!(
            error
                .to_string()
                .contains("Persisted AVL digest is reachable more than once"),
            "unexpected rollback error: {error:#}"
        );
    }

    #[test]
    fn rollback_rejects_a_missing_reachable_descendant_atomically() {
        let dir = tempdir().unwrap();
        let mut storage = RedbAVLStorage::open(&dir.path().join("test.redb"), 32, None).unwrap();
        let negative_key = Bytes::from(vec![0u8; 32]);
        let positive_key = Bytes::from(vec![0xFFu8; 32]);
        let right_key = Bytes::from(vec![1u8; 32]);
        let missing_left = LeafNode::new(&negative_key, &Bytes::new(), &right_key);
        let missing_digest = missing_left.borrow_mut().label();
        let right = LeafNode::new(&right_key, &Bytes::from("right"), &positive_key);
        let root = InternalNode::new(
            Some(right_key),
            &Node::new_label(&missing_digest),
            &right,
            0,
        );
        let target = persist_root(&mut storage, root, 1);

        assert_target_rollback_error_is_atomic(
            &mut storage,
            &target,
            "Reachable AVL node is missing",
        );
    }

    #[test]
    fn rollback_rejects_a_truncated_reachable_descendant_atomically() {
        let dir = tempdir().unwrap();
        let mut storage = RedbAVLStorage::open(&dir.path().join("test.redb"), 32, None).unwrap();
        let (root, left_digest, _) = two_leaf_root(repeated_key(1), repeated_key(1), 0);
        let target = persist_root(&mut storage, root, 1);
        storage
            .store
            .replace_node_for_test(&left_digest, &[1, 0, 0])
            .unwrap();

        assert_target_rollback_error_is_atomic(&mut storage, &target, "Persisted leaf");
    }

    #[test]
    fn rollback_rejects_a_reachable_descendant_with_the_wrong_digest_atomically() {
        let dir = tempdir().unwrap();
        let mut storage = RedbAVLStorage::open(&dir.path().join("test.redb"), 32, None).unwrap();
        let (root, left_digest, _) = two_leaf_root(repeated_key(1), repeated_key(1), 0);
        let target = persist_root(&mut storage, root, 1);
        let replacement = LeafNode::new(
            &repeated_key(2),
            &Bytes::from("replacement"),
            &repeated_key(3),
        );
        let replacement_digest = replacement.borrow_mut().label();
        assert_ne!(replacement_digest, left_digest);
        let tree = AVLTree::new(dummy_resolver, 32, None);
        let replacement_bytes = tree.pack(replacement);
        RedbAVLStorage::validate_packed_node(&replacement_bytes, 32, None).unwrap();
        storage
            .store
            .replace_node_for_test(&left_digest, &replacement_bytes)
            .unwrap();

        assert_target_rollback_error_is_atomic(
            &mut storage,
            &target,
            "Persisted node digest does not match its content-address key",
        );
    }

    #[test]
    fn rollback_rejects_an_unhashed_separator_mutation_atomically() {
        let dir = tempdir().unwrap();
        let mut storage = RedbAVLStorage::open(&dir.path().join("test.redb"), 32, None).unwrap();
        let (root, _, _) = two_leaf_root(repeated_key(1), repeated_key(1), 0);
        let target = persist_root(&mut storage, root, 1);
        let root_digest: Digest32 = target[..32].try_into().unwrap();
        let mut root_bytes = storage.store.get_node(&root_digest).unwrap().unwrap();
        root_bytes[2..34].copy_from_slice(&repeated_key(2));
        storage
            .store
            .replace_node_for_test(&root_digest, &root_bytes)
            .unwrap();
        assert_authentic_record(&storage, &root_digest);

        assert_target_rollback_error_is_atomic(
            &mut storage,
            &target,
            "separator does not equal the minimum key of its right subtree",
        );
    }

    #[test]
    fn rollback_rejects_a_rehashed_broken_leaf_next_chain_atomically() {
        let dir = tempdir().unwrap();
        let mut storage = RedbAVLStorage::open(&dir.path().join("test.redb"), 32, None).unwrap();
        let (root, left_digest, right_digest) = two_leaf_root(repeated_key(1), repeated_key(2), 0);
        let target = persist_root(&mut storage, root, 1);
        let root_digest: Digest32 = target[..32].try_into().unwrap();
        for digest in [root_digest, left_digest, right_digest] {
            assert_authentic_record(&storage, &digest);
        }

        assert_target_rollback_error_is_atomic(
            &mut storage,
            &target,
            "leaf next-key chain is not contiguous",
        );
    }

    #[test]
    fn rollback_rejects_a_rehashed_wrong_balance_atomically() {
        let dir = tempdir().unwrap();
        let mut storage = RedbAVLStorage::open(&dir.path().join("test.redb"), 32, None).unwrap();
        let (root, left_digest, right_digest) = two_leaf_root(repeated_key(1), repeated_key(1), 1);
        let target = persist_root(&mut storage, root, 1);
        let root_digest: Digest32 = target[..32].try_into().unwrap();
        for digest in [root_digest, left_digest, right_digest] {
            assert_authentic_record(&storage, &digest);
        }

        assert_target_rollback_error_is_atomic(
            &mut storage,
            &target,
            "Persisted AVL balance does not match subtree heights",
        );
    }

    #[test]
    fn rollback_rejects_a_wrong_committed_tree_height_atomically() {
        let dir = tempdir().unwrap();
        let mut storage = RedbAVLStorage::open(&dir.path().join("test.redb"), 32, None).unwrap();
        let (root, left_digest, right_digest) = two_leaf_root(repeated_key(1), repeated_key(1), 0);
        let target = persist_root(&mut storage, root, 2);
        let root_digest: Digest32 = target[..32].try_into().unwrap();
        for digest in [root_digest, left_digest, right_digest] {
            assert_authentic_record(&storage, &digest);
        }

        assert_target_rollback_error_is_atomic(
            &mut storage,
            &target,
            "Computed AVL height does not match persisted metadata",
        );
    }

    #[test]
    fn rollback_rejects_a_tree_without_the_negative_infinity_sentinel_atomically() {
        let dir = tempdir().unwrap();
        let mut storage = RedbAVLStorage::open(&dir.path().join("test.redb"), 32, None).unwrap();
        let left = LeafNode::new(&repeated_key(1), &Bytes::from("left"), &repeated_key(2));
        let right = LeafNode::new(&repeated_key(2), &Bytes::from("right"), &repeated_key(0xFF));
        let root = InternalNode::new(Some(repeated_key(2)), &left, &right, 0);
        let target = persist_root(&mut storage, root, 1);
        for digest in [
            <Digest32>::try_from(&target[..32]).unwrap(),
            left.borrow_mut().label(),
            right.borrow_mut().label(),
        ] {
            assert_authentic_record(&storage, &digest);
        }

        assert_target_rollback_error_is_atomic(
            &mut storage,
            &target,
            "missing its negative-infinity sentinel",
        );
    }

    #[test]
    fn rollback_rejects_a_tree_without_the_positive_infinity_sentinel_atomically() {
        let dir = tempdir().unwrap();
        let mut storage = RedbAVLStorage::open(&dir.path().join("test.redb"), 32, None).unwrap();
        let left = LeafNode::new(&repeated_key(0), &Bytes::new(), &repeated_key(1));
        let right = LeafNode::new(&repeated_key(1), &Bytes::from("right"), &repeated_key(0xFE));
        let root = InternalNode::new(Some(repeated_key(1)), &left, &right, 0);
        let target = persist_root(&mut storage, root, 1);
        for digest in [
            <Digest32>::try_from(&target[..32]).unwrap(),
            left.borrow_mut().label(),
            right.borrow_mut().label(),
        ] {
            assert_authentic_record(&storage, &digest);
        }

        assert_target_rollback_error_is_atomic(
            &mut storage,
            &target,
            "missing its positive-infinity sentinel",
        );
    }

    #[test]
    fn rollback_rejects_a_leaf_key_that_does_not_precede_its_next_key_atomically() {
        let dir = tempdir().unwrap();
        let mut storage = RedbAVLStorage::open(&dir.path().join("test.redb"), 32, None).unwrap();
        let invalid_leaf = LeafNode::new(&repeated_key(0), &Bytes::new(), &repeated_key(0));
        let digest = invalid_leaf.borrow_mut().label();
        let target = persist_root(&mut storage, invalid_leaf, 0);
        assert_authentic_record(&storage, &digest);

        assert_target_rollback_error_is_atomic(
            &mut storage,
            &target,
            "leaf key must precede its next key",
        );
    }

    #[test]
    fn open_rejects_a_corrupt_reachable_avl_record() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.redb");
        let mut storage = RedbAVLStorage::open(&path, 32, None).unwrap();
        let (root, left_digest, _) = two_leaf_root(repeated_key(1), repeated_key(1), 0);
        persist_root(&mut storage, root, 1);
        storage
            .store
            .replace_node_for_test(&left_digest, &[1, 0, 0])
            .unwrap();
        drop(storage);

        let reopened = RedbAVLStorage::open(&path, 32, None);

        assert!(reopened.is_err(), "startup must reject corrupt AVL closure");
    }

    #[test]
    fn rollback_returns_a_lazy_multilevel_root_after_full_validation() {
        use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

        let dir = tempdir().unwrap();
        let mut storage = RedbAVLStorage::open(&dir.path().join("test.redb"), 32, None).unwrap();
        let tree = AVLTree::new(dummy_resolver, 32, None);
        let mut prover = BatchAVLProver::new(tree, true);
        storage.update(&mut prover, vec![]).unwrap();
        for byte in 1..=16u8 {
            prover
                .perform_one_operation(&Operation::Insert(KeyValue {
                    key: repeated_key(byte),
                    value: Bytes::from(vec![byte]),
                }))
                .unwrap();
        }
        storage.update(&mut prover, vec![]).unwrap();
        let target = prover.digest().unwrap();

        let validation_resolver_calls = Arc::new(AtomicUsize::new(0));
        let validation_calls = Arc::clone(&validation_resolver_calls);
        let validation_resolver: Resolver = Arc::new(move |digest| {
            validation_calls.fetch_add(1, AtomicOrdering::SeqCst);
            Node::LabelOnly(NodeHeader::new(Some(*digest), None))
        });
        let (root, height) = storage
            .store
            .rollback_with_validation(&target, |lookup| {
                RedbAVLStorage::validate_and_load_snapshot(
                    lookup,
                    &target,
                    32,
                    None,
                    validation_resolver,
                )
            })
            .unwrap();
        assert_eq!(validation_resolver_calls.load(AtomicOrdering::SeqCst), 0);

        let resolver_calls = Arc::new(AtomicUsize::new(0));
        let calls = Arc::clone(&resolver_calls);
        let persistent_resolver = storage.create_resolver();
        let counting_resolver: Resolver = Arc::new(move |digest| {
            calls.fetch_add(1, AtomicOrdering::SeqCst);
            persistent_resolver(digest)
        });
        assert_eq!(resolver_calls.load(AtomicOrdering::SeqCst), 0);
        match &*root.borrow() {
            Node::Internal(internal) => {
                assert!(matches!(&*internal.left.borrow(), Node::LabelOnly(_)));
                assert!(matches!(&*internal.right.borrow(), Node::LabelOnly(_)));
            }
            _ => panic!("multi-level persisted root must be internal"),
        }

        let mut restored_tree = AVLTree::with_resolver(counting_resolver, 32, None);
        restored_tree.root = Some(root);
        restored_tree.height = height;
        let restored = BatchAVLProver::new(restored_tree, true);
        assert_eq!(
            restored.unauthenticated_lookup(&repeated_key(8)),
            Some(Bytes::from(vec![8u8]))
        );
        assert!(resolver_calls.load(AtomicOrdering::SeqCst) > 0);
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
    fn open_rejects_zero_length_avl_keys() {
        let dir = tempdir().unwrap();

        let error = match RedbAVLStorage::open(&dir.path().join("test.redb"), 0, None) {
            Ok(_) => panic!("zero-length AVL keys must be rejected"),
            Err(error) => error,
        };

        assert_eq!(
            error.to_string(),
            "AVL key length must be greater than zero"
        );
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

    #[test]
    fn malformed_packed_nodes_are_errors_not_panics() {
        let (tree, leaf, digest) = packed_leaf();
        for end in 0..leaf.len() {
            let truncated = Bytes::copy_from_slice(&leaf[..end]);
            let outcome = catch_unwind(AssertUnwindSafe(|| {
                RedbAVLStorage::unpack_verified_node(&tree, &truncated, &digest)
            }));
            assert!(outcome.is_ok(), "packed-node truncation must not panic");
            assert!(
                outcome.unwrap().is_err(),
                "packed-node truncation must fail"
            );
        }

        let mut trailing = leaf.to_vec();
        trailing.push(0);
        let trailing = Bytes::from(trailing);
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            RedbAVLStorage::unpack_verified_node(&tree, &trailing, &digest)
        }));
        assert!(outcome.is_ok(), "packed-node trailing bytes must not panic");
        assert!(
            outcome.unwrap().is_err(),
            "packed-node trailing bytes must fail"
        );

        for malformed in [vec![2], vec![0, 2]] {
            let malformed = Bytes::from(malformed);
            let outcome = catch_unwind(AssertUnwindSafe(|| {
                RedbAVLStorage::unpack_verified_node(&tree, &malformed, &digest)
            }));
            assert!(outcome.is_ok(), "malformed packed node must not panic");
            assert!(outcome.unwrap().is_err(), "malformed packed node must fail");
        }
    }

    fn assert_packed_err_without_panic(
        bytes: &[u8],
        key_length: usize,
        value_length: Option<usize>,
    ) {
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            RedbAVLStorage::validate_packed_node(bytes, key_length, value_length)
        }));
        assert!(outcome.is_ok(), "packed-node validation must not panic");
        assert!(outcome.unwrap().is_err(), "malformed packed node must fail");
    }

    #[test]
    fn packed_node_validator_is_canonical_for_every_layout() {
        let mut internal = vec![0, 0];
        internal.extend_from_slice(&[7; 4]);
        internal.extend_from_slice(&[8; 32]);
        internal.extend_from_slice(&[9; 32]);

        let mut fixed_leaf = vec![1];
        fixed_leaf.extend_from_slice(&[1; 4]);
        fixed_leaf.extend_from_slice(&[2; 3]);
        fixed_leaf.extend_from_slice(&[3; 4]);

        let mut variable_leaf = vec![1];
        variable_leaf.extend_from_slice(&[1; 4]);
        variable_leaf.extend_from_slice(&3u32.to_be_bytes());
        variable_leaf.extend_from_slice(&[2; 3]);
        variable_leaf.extend_from_slice(&[3; 4]);

        for (bytes, key_length, value_length) in [
            (&internal[..], 4, None),
            (&fixed_leaf[..], 4, Some(3)),
            (&variable_leaf[..], 4, None),
        ] {
            RedbAVLStorage::validate_packed_node(bytes, key_length, value_length).unwrap();
            for end in 0..bytes.len() {
                assert_packed_err_without_panic(&bytes[..end], key_length, value_length);
            }
            let mut trailing = bytes.to_vec();
            trailing.push(0);
            assert_packed_err_without_panic(&trailing, key_length, value_length);
        }

        for balance in [2u8, 254u8] {
            let mut invalid = internal.clone();
            invalid[1] = balance;
            assert_packed_err_without_panic(&invalid, 4, None);
        }

        for declared in [2u32, 4, u32::MAX] {
            let mut invalid = variable_leaf.clone();
            invalid[5..9].copy_from_slice(&declared.to_be_bytes());
            assert_packed_err_without_panic(&invalid, 4, None);
        }

        assert_packed_err_without_panic(&[0], usize::MAX, None);
        assert_packed_err_without_panic(&[1], usize::MAX, None);
        assert_packed_err_without_panic(&[1], 0, Some(usize::MAX));
    }

    #[test]
    fn content_addressed_dedup_rejects_same_digest_with_different_internal_key() {
        let tree = AVLTree::new(dummy_resolver, 32, None);
        let left_digest = [4u8; 32];
        let right_digest = [5u8; 32];
        let left = Node::new_label(&left_digest);
        let right = Node::new_label(&right_digest);
        let first = InternalNode::new(Some(Bytes::from(vec![1u8; 32])), &left, &right, 0);
        let second = InternalNode::new(Some(Bytes::from(vec![2u8; 32])), &left, &right, 0);
        let digest = first.borrow_mut().label();
        assert_eq!(second.borrow_mut().label(), digest);
        let first_bytes = tree.pack(first);
        let second_bytes = tree.pack(second);
        assert_ne!(first_bytes, second_bytes);

        let dir = tempdir().unwrap();
        let store = RedbVersionedStore::open(&dir.path().join("test.redb")).unwrap();
        store
            .update_partitioned(
                b"v1",
                &[(digest.as_slice(), first_bytes.as_ref())],
                &[],
                &[],
            )
            .unwrap();

        let error = store
            .update_partitioned(
                b"v2",
                &[(digest.as_slice(), second_bytes.as_ref())],
                &[],
                &[],
            )
            .unwrap_err();

        assert_eq!(
            error.to_string(),
            "Content-addressed node key conflicts with different stored bytes"
        );
        assert_eq!(store.get_node(&digest).unwrap(), Some(first_bytes.to_vec()));
        assert_eq!(store.last_version_id().unwrap(), Some(b"v1".to_vec()));
        assert_eq!(store.rollback_versions().unwrap(), vec![b"v1".to_vec()]);
    }

    #[test]
    fn corrupt_target_root_aborts_rollback_atomically() {
        let dir = tempdir().unwrap();
        let mut storage = RedbAVLStorage::open(&dir.path().join("test.redb"), 32, None).unwrap();
        let tree = AVLTree::new(dummy_resolver, 32, None);
        let mut prover = BatchAVLProver::new(tree, true);

        storage.update(&mut prover, vec![]).unwrap();
        let target = prover.digest().unwrap();
        let target_root = storage.store.get_node(TOP_NODE_HASH_KEY).unwrap().unwrap();

        let tip = [9u8; 33];
        storage
            .store
            .update(&tip, &[(b"dummy", b"value")], &[])
            .unwrap();
        let tip_root = storage.store.get_node(TOP_NODE_HASH_KEY).unwrap().unwrap();
        let versions_before = storage.store.rollback_versions().unwrap();

        storage
            .store
            .replace_node_for_test(&target_root, &[1, 2, 3])
            .unwrap();
        let error = storage.rollback(&target).unwrap_err();

        assert!(error.to_string().contains("Persisted leaf"));
        assert_eq!(storage.store.last_version_id().unwrap(), Some(tip.to_vec()));
        assert_eq!(storage.store.rollback_versions().unwrap(), versions_before);
        assert_eq!(
            storage.store.get_node(b"dummy").unwrap(),
            Some(b"value".to_vec())
        );
        assert_eq!(
            storage.store.get_node(TOP_NODE_HASH_KEY).unwrap(),
            Some(tip_root)
        );
    }

    #[test]
    fn malformed_target_height_aborts_rollback_atomically() {
        let dir = tempdir().unwrap();
        let mut storage = RedbAVLStorage::open(&dir.path().join("test.redb"), 32, None).unwrap();
        let tree = AVLTree::new(dummy_resolver, 32, None);
        let mut prover = BatchAVLProver::new(tree, true);
        storage.update(&mut prover, vec![]).unwrap();
        let target = prover.digest().unwrap();

        let tip = [9u8; 33];
        storage
            .store
            .update(&tip, &[(b"dummy", b"value")], &[])
            .unwrap();
        let tip_root = storage.store.get_node(TOP_NODE_HASH_KEY).unwrap().unwrap();
        let versions_before = storage.store.rollback_versions().unwrap();
        storage
            .store
            .replace_node_for_test(TOP_NODE_HEIGHT_KEY, &[0u8; 7])
            .unwrap();

        let error = storage.rollback(&target).unwrap_err();

        assert_eq!(
            error.to_string(),
            "Persisted height must be exactly 8 bytes"
        );
        assert_eq!(storage.store.last_version_id().unwrap(), Some(tip.to_vec()));
        assert_eq!(storage.store.rollback_versions().unwrap(), versions_before);
        assert_eq!(
            storage.store.get_node(b"dummy").unwrap(),
            Some(b"value".to_vec())
        );
        assert_eq!(
            storage.store.get_node(TOP_NODE_HASH_KEY).unwrap(),
            Some(tip_root)
        );
    }

    #[test]
    fn fallible_history_query_surfaces_corruption_and_legacy_query_fails_stop() {
        let dir = tempdir().unwrap();
        let storage = RedbAVLStorage::open(&dir.path().join("test.redb"), 32, None).unwrap();
        storage
            .store
            .update(b"v1", &[(b"key", b"one")], &[])
            .unwrap();
        storage
            .store
            .update(b"v2", &[(b"key", b"two")], &[])
            .unwrap();

        // Remove v1 while retaining v2 -> v1 to model a corrupt persisted chain.
        storage.store.remove_version_for_test(b"v1").unwrap();

        assert!(storage.try_rollback_versions().is_err());
        let legacy = catch_unwind(AssertUnwindSafe(|| {
            let _ = storage.rollback_versions().collect::<Vec<_>>();
        }));
        assert!(legacy.is_err(), "legacy query must fail-stop on corruption");
    }

    #[test]
    fn fallible_version_and_history_queries_reject_partial_metadata() {
        let dir = tempdir().unwrap();
        let storage = RedbAVLStorage::open(&dir.path().join("test.redb"), 32, None).unwrap();
        storage
            .store
            .set_metadata_for_test(Some(b"v1"), None)
            .unwrap();
        assert!(storage.try_version().is_err());
        assert!(storage.try_rollback_versions().is_err());

        let next = 0u64.to_le_bytes();
        storage
            .store
            .set_metadata_for_test(None, Some(&next))
            .unwrap();
        assert!(storage.try_version().is_err());
        assert!(storage.try_rollback_versions().is_err());
    }
}
