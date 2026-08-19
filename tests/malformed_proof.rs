//! Regression tests: malformed, truncated, or empty proofs — and out-of-range
//! tree parameters — must fail gracefully (returning an `Err`) rather than
//! panicking with a slice-bounds, stack-underflow, or arithmetic-overflow
//! panic. This mirrors the reference (scorex) `BatchAVLVerifier`, whose tree
//! reconstruction is wrapped in a `Try`: any failure simply yields a verifier
//! with no reconstructed tree, and subsequent operations then fail cleanly.

use bytes::Bytes;
use ergo_avltree_rust::authenticated_tree_ops::AuthenticatedTreeOps;
use ergo_avltree_rust::batch_avl_verifier::BatchAVLVerifier;
use ergo_avltree_rust::operation::*;

mod common;
use common::*;

// Proof structure opcodes (see `batch_node`): a leaf node, a label-only node,
// and the end-of-tree marker, respectively. Referenced by value here because
// they are crate-private.
const LEAF: u8 = 2;
const LABEL: u8 = 3;

/// A syntactically well-formed 33-byte digest: 32-byte root hash + height byte.
fn dummy_digest() -> Bytes {
    Bytes::from(vec![7u8; 33])
}

#[test]
fn empty_proof_is_err_not_panic() {
    // Used to panic indexing `self.proof[0]` on an empty proof.
    let v = BatchAVLVerifier::new(
        &dummy_digest(),
        &Bytes::new(),
        generate_tree(KEY_LENGTH, None),
        None,
        None,
    );
    assert!(v.is_err());
}

#[test]
fn internal_node_opcode_with_empty_stack_is_err_not_panic() {
    // A single non-leaf/non-label opcode reaches the internal-node arm and used
    // to panic on `stack.pop().unwrap()` with nothing on the stack.
    let v = BatchAVLVerifier::new(
        &dummy_digest(),
        &Bytes::from(vec![0u8]),
        generate_tree(KEY_LENGTH, None),
        None,
        None,
    );
    assert!(v.is_err());
}

#[test]
fn truncated_label_proof_is_err_not_panic() {
    // A label opcode promises 32 following bytes; with none present the digest
    // slice used to panic out of bounds.
    let v = BatchAVLVerifier::new(
        &dummy_digest(),
        &Bytes::from(vec![LABEL]),
        generate_tree(KEY_LENGTH, None),
        None,
        None,
    );
    assert!(v.is_err());
}

#[test]
fn truncated_real_proof_is_err_not_panic() {
    // A valid proof cut in half: a genuine reconstruction that runs out of bytes
    // partway through, rather than a hand-crafted shape.
    let (mut prover, _) = generate_and_populate_prover(16);
    let proof = prover.generate_proof();
    let digest = prover.digest().unwrap();
    let half = Bytes::copy_from_slice(&proof[..proof.len() / 2]);
    let v = BatchAVLVerifier::new(&digest, &half, generate_tree(KEY_LENGTH, None), None, None);
    assert!(v.is_err());
}

#[test]
fn oversized_key_length_is_err_not_panic() {
    // A wrapped/out-of-range key length used to overflow `i + key_length` or
    // slice the proof out of bounds while reading the first leaf key.
    let v = BatchAVLVerifier::new(
        &dummy_digest(),
        &Bytes::from(vec![LEAF]),
        generate_tree(usize::MAX, None),
        None,
        None,
    );
    assert!(v.is_err());
}

#[test]
fn wrong_value_length_operation_is_err_not_panic() {
    // A fixed-value-length tree: performing an operation whose value length does
    // not match used to fire `assert!(value_length matches)` inside the op.
    let mut prover = generate_prover(KEY_LENGTH, Some(VALUE_LENGTH));
    let initial_digest = prover.digest().unwrap();
    let key = Bytes::from(vec![1u8; KEY_LENGTH]);
    prover
        .perform_one_operation(&Operation::Insert(KeyValue {
            key: key.clone(),
            value: Bytes::from(vec![0u8; VALUE_LENGTH]),
        }))
        .unwrap();
    let proof = prover.generate_proof();

    let mut verifier = BatchAVLVerifier::new(
        &initial_digest,
        &proof,
        generate_tree(KEY_LENGTH, Some(VALUE_LENGTH)),
        None,
        None,
    )
    .unwrap();
    // Replay the same insert, but with a value whose length differs from the
    // tree's fixed value length.
    let res = verifier.perform_one_operation(&Operation::Insert(KeyValue {
        key,
        value: Bytes::from(vec![0u8; VALUE_LENGTH + 1]),
    }));
    assert!(res.is_err());
}
