#![cfg(all(not(target_arch = "wasm32"), not(miri)))]

use bytes::Bytes;
use ergo_avltree_rust::authenticated_tree_ops::AuthenticatedTreeOps;
use ergo_avltree_rust::batch_avl_verifier::BatchAVLVerifier;
use ergo_avltree_rust::batch_node::{Node, NodeId};
use std::env;
use std::mem;
use std::process::Command;
use std::rc::Rc;
use std::thread;

mod common;
#[path = "common/deep_spine.rs"]
mod deep_spine;

use common::generate_tree;
use deep_spine::{
    assert_left_spine_depth, deep_spine_proof, hash_matching_digest, DEEP_DROP_DEPTH,
};

const CHILD_CASE_ENV: &str = "ERGO_AVLTREE_DROP_CHILD";
const WORKER_STACK: usize = 512 * 1024;
const INTERIOR_DEPTH: usize = DEEP_DROP_DEPTH / 2;

const CASES: &[&str] = &[
    "invalid-digest",
    "forget",
    "direct-drop",
    "top-outlives-verifier",
    "verifier-outlives-top",
    "tree-outlives-verifier",
    "verifier-outlives-tree",
    "shared-interior",
    "borrowed-interior",
];

fn verifier() -> BatchAVLVerifier {
    BatchAVLVerifier::new(
        &hash_matching_digest(DEEP_DROP_DEPTH, 0),
        &deep_spine_proof(DEEP_DROP_DEPTH, false),
        generate_tree(1, Some(1)),
        None,
        None,
    )
    .expect("hash-matching deep spine must construct")
}

fn left_spine_node(root: &NodeId, depth: usize) -> NodeId {
    let mut current = root.clone();
    for _ in 0..depth {
        let next = match &*current.borrow() {
            Node::Internal(node) => node.left.clone(),
            _ => panic!("left spine ended before interior node"),
        };
        current = next;
    }
    current
}

fn run_child_case(case: &str) {
    match case {
        "invalid-digest" => {
            let invalid = BatchAVLVerifier::new(
                &Bytes::from(vec![7u8; 33]),
                &deep_spine_proof(DEEP_DROP_DEPTH, false),
                generate_tree(1, Some(1)),
                None,
                None,
            );
            assert!(invalid.is_err(), "invalid digest must be rejected");
        }
        "forget" => mem::forget(verifier()),
        "direct-drop" => drop(verifier()),
        "top-outlives-verifier" => {
            let verifier = verifier();
            let top = verifier.top_node();
            drop(verifier);
            assert_left_spine_depth(&top, DEEP_DROP_DEPTH);
            drop(top);
        }
        "verifier-outlives-top" => {
            let verifier = verifier();
            let top = verifier.top_node();
            assert_left_spine_depth(&top, DEEP_DROP_DEPTH);
            drop(top);
            drop(verifier);
        }
        "tree-outlives-verifier" => {
            let verifier = verifier();
            let tree = verifier.get_tree().clone();
            drop(verifier);
            let root = tree.root.as_ref().expect("verified tree must have a root");
            assert_left_spine_depth(root, DEEP_DROP_DEPTH);
            drop(tree);
        }
        "verifier-outlives-tree" => {
            let verifier = verifier();
            let tree = verifier.get_tree().clone();
            let root = tree.root.as_ref().expect("verified tree must have a root");
            assert_left_spine_depth(root, DEEP_DROP_DEPTH);
            drop(tree);
            drop(verifier);
        }
        "shared-interior" => {
            let verifier = verifier();
            let shared = left_spine_node(&verifier.top_node(), INTERIOR_DEPTH);
            assert_eq!(Rc::strong_count(&shared), 2);
            drop(verifier);
            assert_eq!(Rc::strong_count(&shared), 1);
            assert_left_spine_depth(&shared, DEEP_DROP_DEPTH - INTERIOR_DEPTH);
            drop(shared);
        }
        "borrowed-interior" => {
            let verifier = verifier();
            let interior = left_spine_node(&verifier.top_node(), INTERIOR_DEPTH);
            let borrow = interior.borrow();
            assert!(matches!(&*borrow, Node::Internal(_)));
            assert_eq!(Rc::strong_count(&interior), 2);
            drop(verifier);
            assert_eq!(Rc::strong_count(&interior), 1);
            assert!(matches!(&*borrow, Node::Internal(_)));
            drop(borrow);
            assert_left_spine_depth(&interior, DEEP_DROP_DEPTH - INTERIOR_DEPTH);
            drop(interior);
        }
        other => panic!("unknown iterative-drop child case: {other}"),
    }
}

#[test]
fn iterative_drop_child() {
    let Ok(case) = env::var(CHILD_CASE_ENV) else {
        return;
    };

    let worker = thread::Builder::new()
        .stack_size(WORKER_STACK)
        .spawn(move || run_child_case(&case))
        .expect("spawn teardown worker");
    worker.join().expect("teardown worker panicked");
}

#[test]
fn iterative_drop_matrix_is_stack_safe() {
    for case in CASES {
        let status = Command::new(env::current_exe().expect("test executable path"))
            .env(CHILD_CASE_ENV, case)
            .arg("--exact")
            .arg("iterative_drop_child")
            .status()
            .expect("launch teardown child");
        assert!(status.success(), "teardown child failed for {case}");
    }
}
