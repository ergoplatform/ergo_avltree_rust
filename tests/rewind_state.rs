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

use ergo_avltree_rust::batch_avl_prover::BatchAVLProver;
use ergo_avltree_rust::operation::*;
use ergo_avltree_rust::persistent_batch_avl_prover::*;

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
