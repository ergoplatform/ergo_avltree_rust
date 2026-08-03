//! Regression test: a crafted proof encoding a deep left spine must not drive
//! label computation to the packed depth.
//!
//! `Node::label` used to recurse into both children, so the recursion depth was
//! the depth of the tree — and a verifier's tree is materialised from proof
//! bytes. `reconstruct_tree` evaluates `label(&root)` *before* comparing it
//! against the starting digest, so the attacker needs no valid digest at all:
//! any 33 bytes reach the traversal. The result was a stack overflow, which
//! aborts the process rather than raising a catchable panic.
//!
//! The work runs on a worker with an explicit stack size, so the test measures
//! the implementation rather than the platform's default thread stack — an
//! earlier version keyed off the default and passed at 2 MiB while aborting at
//! 512 KiB.
//!
//! Both bounds below are measured, not guessed:
//!
//! * the recursive `label` exhausted this worker at roughly depth 700, so 2,000
//!   fails loudly if the iterative walk regresses (verified by reverting it);
//! * iterative last-owner teardown has its own 50,000-level subprocess matrix
//!   in `iterative_drop.rs`, so this 2,000-level test remains focused on label
//!   computation rather than destructor behavior.
//!
//! A regression aborts the process rather than failing the assertion — that is
//! inherent to testing stack exhaustion, and the abort is the signal.

use bytes::Bytes;
use ergo_avltree_rust::batch_avl_verifier::BatchAVLVerifier;
use ergo_avltree_rust::operation::Operation;
use std::thread;

mod common;
#[path = "common/deep_spine.rs"]
mod deep_spine;
use common::generate_tree;
use deep_spine::{deep_spine_proof, hash_matching_digest, LEAF_VALUE, LOOKUP_KEY};

/// Explicit worker stack: small enough to be a meaningful bound, large enough
/// that the depth below is not near the teardown limit.
const WORKER_STACK: usize = 1 << 20; // 1 MiB

/// ~68 KB of proof. See the module comment for why this depth.
const SPINE_DEPTH: usize = 2_000;

#[test]
fn deep_spine_proof_is_rejected_without_exhausting_the_stack() {
    let worker = thread::Builder::new()
        .stack_size(WORKER_STACK)
        .spawn(|| {
            // 33 arbitrary bytes: the digest comparison is never reached, the
            // traversal that computes the label to compare against it is.
            let v = BatchAVLVerifier::new(
                &Bytes::from(vec![7u8; 33]),
                &deep_spine_proof(SPINE_DEPTH, false),
                generate_tree(1, Some(1)),
                None,
                None,
            );
            assert!(v.is_err(), "deep-spine proof must be rejected");
        })
        .expect("spawn worker");

    worker.join().expect("worker panicked");
}

#[test]
fn deep_spine_lookup_accepts_boundary_depths() {
    for depth in [255, 256, 300] {
        let mut verifier = BatchAVLVerifier::new(
            &hash_matching_digest(depth, 0),
            &deep_spine_proof(depth, true),
            generate_tree(1, Some(1)),
            None,
            None,
        )
        .expect("hash-matching deep spine must construct");

        assert_eq!(
            verifier
                .perform_one_operation(&Operation::Lookup(Bytes::copy_from_slice(&LOOKUP_KEY)))
                .expect("lookup must verify"),
            Some(Bytes::copy_from_slice(&LEAF_VALUE)),
            "depth {depth}"
        );
    }
}
