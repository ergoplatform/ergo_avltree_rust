//! A rewind must abandon the whole proof cycle, not part of it.
//!
//! `restore_root` and `PersistentBatchAVLProver::rollback` both discard an
//! in-flight cycle. Everything that cycle accumulated has to go with it —
//! including `modified_nodes`, which `pack_tree` gates on. `generate_proof`
//! clears that map, but a cycle that is abandoned never reaches it, so the
//! rewind is the only place left to do it.

// `common` is shared by every integration test; each binary compiles the whole
// module and warns about the helpers it happens not to use.
#[allow(dead_code)]
mod common;
use common::*;

use bytes::Bytes;
use ergo_avltree_rust::authenticated_tree_ops::AuthenticatedTreeOps;
use ergo_avltree_rust::batch_avl_prover::BatchAVLProver;
use ergo_avltree_rust::batch_node::{AVLTree, InternalNode, LeafNode, Node, NodeId};
use ergo_avltree_rust::operation::*;
use ergo_avltree_rust::persistent_batch_avl_prover::*;
use std::cell::RefCell;
use std::rc::Rc;

thread_local! {
    static LAZY_RESOLVER_CACHE: RefCell<Option<Node>> = RefCell::new(None);
}

/// Applies `kvs` as inserts and stops there, without closing the proof cycle —
/// the shape of a block that is applied and then rejected.
fn apply_without_proof(prover: &mut BatchAVLProver, kvs: &[KeyValue]) {
    for kv in kvs {
        prover
            .perform_one_operation(&Operation::Insert(kv.clone()))
            .unwrap();
    }
}

/// Distinct deterministic key sets, so the two provers in the round-trip test
/// are driven identically.
fn kv_batch(offset: usize, size: usize) -> Vec<KeyValue> {
    generate_kv_list(offset + size)[offset..].to_vec()
}

fn node_flags(node: &NodeId, flags: &mut Vec<(usize, bool, bool)>) {
    let node_ref = node.borrow();
    flags.push((
        Rc::as_ptr(node) as usize,
        node_ref.visited(),
        node_ref.is_new(),
    ));
    let children = match &*node_ref {
        Node::Internal(internal) => Some((internal.left.clone(), internal.right.clone())),
        _ => None,
    };
    drop(node_ref);

    if let Some((left, right)) = children {
        node_flags(&left, flags);
        node_flags(&right, flags);
    }
}

fn node_ids(nodes: &[NodeId]) -> Vec<usize> {
    nodes.iter().map(|node| Rc::as_ptr(node) as usize).collect()
}

fn lazy_resolver(_digest: &Digest32) -> Node {
    LAZY_RESOLVER_CACHE.with(|cache| cache.borrow().as_ref().unwrap().clone())
}

fn lazy_cache_snapshot() -> (bool, Vec<(usize, bool, bool)>) {
    LAZY_RESOLVER_CACHE.with(|cache| {
        let cache = cache.borrow();
        let node = cache.as_ref().unwrap();
        let mut flags = Vec::new();
        if let Node::Internal(internal) = node {
            node_flags(&internal.left, &mut flags);
            node_flags(&internal.right, &mut flags);
        }
        (node.is_internal(), flags)
    })
}

#[test]
fn restore_root_clears_modified_nodes() {
    let mut prover = generate_prover(KEY_LENGTH, None);
    apply_without_proof(&mut prover, &kv_batch(0, 10));
    prover.generate_proof();

    let root = prover.base.tree.root.clone().unwrap();
    let height = prover.base.tree.height;

    // An abandoned cycle: applied, never proved.
    apply_without_proof(&mut prover, &kv_batch(100, 10));
    assert!(
        !prover.base.modified_nodes.is_empty(),
        "precondition: the abandoned cycle must have populated modified_nodes"
    );

    prover.restore_root(root, height);

    assert!(
        prover.base.modified_nodes.is_empty(),
        "restore_root left {} entries in modified_nodes; an abandoned cycle's \
         node set must not survive the rewind",
        prover.base.modified_nodes.len()
    );
}

#[test]
fn rollback_clears_modified_nodes() {
    let storage = Box::new(VersionedAVLStorageMock::new());
    let mut prover =
        PersistentBatchAVLProver::new(generate_prover(KEY_LENGTH, None), storage, Vec::new())
            .unwrap();

    apply_without_proof(&mut prover.prover, &kv_batch(0, 10));
    prover
        .generate_proof_and_update_storage(Vec::new())
        .unwrap();
    let version = prover.digest();

    apply_without_proof(&mut prover.prover, &kv_batch(100, 10));
    assert!(
        !prover.prover.base.modified_nodes.is_empty(),
        "precondition: the abandoned cycle must have populated modified_nodes"
    );

    prover.rollback(&version).unwrap();

    assert!(
        prover.prover.base.modified_nodes.is_empty(),
        "rollback left {} entries in modified_nodes; it is a rewind and must \
         drop the abandoned cycle's state like restore_root does",
        prover.prover.base.modified_nodes.len()
    );
}

/// The consequence, not just the bookkeeping: `pack_tree` expands any node in
/// `modified_nodes` instead of emitting its label. A rewind that leaves the
/// previous cycle's nodes marked therefore produces a *different proof* for an
/// identical tree state — a divergence, not merely retained memory.
#[test]
fn proof_after_rejected_cycle_matches_uncontaminated_prover() {
    let base = kv_batch(0, 10);
    let rejected = kv_batch(100, 10);
    let next = kv_batch(200, 10);

    // Contaminated: applies a cycle that gets rejected, rewinds, then proceeds.
    let mut a = PersistentBatchAVLProver::new(
        generate_prover(KEY_LENGTH, None),
        Box::new(VersionedAVLStorageMock::new()),
        Vec::new(),
    )
    .unwrap();
    apply_without_proof(&mut a.prover, &base);
    a.generate_proof_and_update_storage(Vec::new()).unwrap();
    let version = a.digest();
    apply_without_proof(&mut a.prover, &rejected);
    a.rollback(&version).unwrap();
    apply_without_proof(&mut a.prover, &next);
    let proof_a = a.generate_proof_and_update_storage(Vec::new()).unwrap();

    // Clean: same states, no rejected cycle in between.
    let mut b = PersistentBatchAVLProver::new(
        generate_prover(KEY_LENGTH, None),
        Box::new(VersionedAVLStorageMock::new()),
        Vec::new(),
    )
    .unwrap();
    apply_without_proof(&mut b.prover, &base);
    b.generate_proof_and_update_storage(Vec::new()).unwrap();
    apply_without_proof(&mut b.prover, &next);
    let proof_b = b.generate_proof_and_update_storage(Vec::new()).unwrap();

    assert_eq!(
        a.digest(),
        b.digest(),
        "test is malformed if the two provers do not reach the same state"
    );
    assert_eq!(
        proof_a,
        proof_b,
        "a rewound prover produced a different proof ({} bytes) than an \
         uncontaminated one ({} bytes) for identical state",
        proof_a.len(),
        proof_b.len()
    );
}

#[test]
fn preview_keeps_an_in_flight_persistence_cycle_unchanged() {
    let mut prover = generate_prover(KEY_LENGTH, Some(8));
    let applied = KeyValue {
        key: Bytes::from(vec![0x11; KEY_LENGTH]),
        value: Bytes::from(vec![0xAA; 8]),
    };
    let preview = Operation::Insert(KeyValue {
        key: Bytes::from(vec![0x22; KEY_LENGTH]),
        value: Bytes::from(vec![0xBB; 8]),
    });
    prover
        .perform_one_operation(&Operation::Insert(applied.clone()))
        .unwrap();

    let root_before = prover.top_node();
    let digest_before = prover.digest().unwrap();
    let mut flags_before = Vec::new();
    node_flags(&root_before, &mut flags_before);
    let changed_before = node_ids(&prover.base.changed_nodes_buffer);
    let changed_to_check_before = node_ids(&prover.base.changed_nodes_buffer_to_check);
    let modified_before: Vec<usize> = prover.base.modified_nodes.keys().copied().collect();

    prover
        .generate_proof_for_operations(&vec![preview.clone()])
        .unwrap();

    let root_after = prover.top_node();
    let mut flags_after = Vec::new();
    node_flags(&root_after, &mut flags_after);
    assert!(Rc::ptr_eq(&root_before, &root_after));
    assert_eq!(prover.digest().unwrap(), digest_before);
    assert_eq!(flags_after, flags_before);
    assert_eq!(node_ids(&prover.base.changed_nodes_buffer), changed_before);
    assert_eq!(
        node_ids(&prover.base.changed_nodes_buffer_to_check),
        changed_to_check_before
    );
    assert_eq!(
        prover.base.modified_nodes.keys().copied().collect::<Vec<_>>(),
        modified_before
    );
    assert_eq!(prover.unauthenticated_lookup(&applied.key), Some(applied.value));
    assert!(prover.unauthenticated_lookup(&preview.key()).is_none());

    let mut proof_prover = generate_prover(KEY_LENGTH, Some(8));
    let (preview_proof, preview_digest) = proof_prover
        .generate_proof_for_operations(&vec![preview.clone()])
        .unwrap();
    proof_prover.perform_one_operation(&preview).unwrap();
    assert_eq!(proof_prover.generate_proof(), preview_proof);
    assert_eq!(proof_prover.digest().unwrap(), preview_digest);
}

#[test]
fn preview_does_not_mutate_lazy_resolver_cache_children() {
    let min = Bytes::from(vec![0x00; KEY_LENGTH]);
    let split = Bytes::from(vec![0x40; KEY_LENGTH]);
    let root_key = Bytes::from(vec![0x80; KEY_LENGTH]);
    let max = Bytes::from(vec![0xFF; KEY_LENGTH]);
    let value = Bytes::from(vec![0x00; 8]);
    let lazy_left = LeafNode::new(&min, &value, &split);
    let lazy_right = LeafNode::new(&split, &value, &root_key);
    let lazy_internal = InternalNode::new(Some(split.clone()), &lazy_left, &lazy_right, 0);
    LAZY_RESOLVER_CACHE.with(|cache| cache.replace(Some(lazy_internal.borrow().clone())));

    let lazy_label = Node::new_label(&[0xA5; 32]);
    let right = LeafNode::new(&root_key, &value, &max);
    let root = InternalNode::new(Some(root_key.clone()), &lazy_label, &right, -1);
    let mut tree = AVLTree::new(lazy_resolver, KEY_LENGTH, Some(8));
    tree.root = Some(root);
    tree.height = 2;
    let prover = BatchAVLProver::new(tree, true);
    prover.base.tree.reset();

    let cache_before = lazy_cache_snapshot();
    let operation = Operation::Insert(KeyValue {
        key: Bytes::from(vec![0x10; KEY_LENGTH]),
        value: Bytes::from(vec![0xCC; 8]),
    });
    prover.generate_proof_for_operations(&vec![operation]).unwrap();

    assert_eq!(lazy_cache_snapshot(), cache_before);
}
