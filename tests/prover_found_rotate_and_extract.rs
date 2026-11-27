use bytes::Bytes;
use ergo_avltree_rust::authenticated_tree_ops::AuthenticatedTreeOps;
use ergo_avltree_rust::batch_avl_prover::BatchAVLProver;
use ergo_avltree_rust::batch_node::{AVLTree, InternalNode, LeafNode, Node, NodeHeader};
use ergo_avltree_rust::operation::{KeyValue, Operation};
use std::cell::Cell;
use std::rc::Rc;

fn key(byte: u8) -> Bytes {
    Bytes::from(vec![byte; 32])
}

fn val(byte: u8) -> Bytes {
    Bytes::from(vec![byte; 8])
}

fn unresolvable(digest: &[u8; 32]) -> Node {
    Node::LabelOnly(NodeHeader::new(Some(*digest), None))
}

/// Regression for sticky `BatchAVLProver::found` after a mid-descent error:
/// when `key == r.key` sets `self.found = true` and descent into `r.right`
/// fails on an unresolvable `LabelOnly`, the next operation must NOT be
/// steered onto the `-inf` sentinel leaf.
#[test]
fn prover_resets_found_after_mid_descent_error() {
    let mut prover = BatchAVLProver::new(AVLTree::new(unresolvable, 32, Some(8)), true);
    for b in [0x10u8, 0x20, 0x30, 0x40] {
        prover
            .perform_one_operation(&Operation::Insert(KeyValue {
                key: key(b),
                value: val(b),
            }))
            .unwrap();
    }
    prover.generate_proof();

    let root = prover.base.tree.root.clone().unwrap();
    let (root_key, right_label) = {
        let borrowed = root.borrow();
        match &*borrowed {
            Node::Internal(r) => (r.hdr.key.clone().unwrap(), prover.base.tree.label(&r.right)),
            _ => panic!("expected internal root"),
        }
    };

    // Unload the root's right subtree to an unresolvable placeholder.
    if let Node::Internal(r) = &mut *root.borrow_mut() {
        r.right = Node::new_label(&right_label);
    }

    // Looking up `root_key` matches `r.key` (setting `self.found = true`) and
    // then descends into `r.right`, which fails on the unresolvable `LabelOnly`.
    let err = prover.perform_one_operation(&Operation::Lookup(root_key));
    assert!(err.is_err(), "lookup into unloaded right subtree must fail");

    // Next operation on an absent key in the materialized left subtree (below 0x10):
    // if `self.found` remained sticky, `next_direction_is_left` would route to `-inf`
    // and `key_matches_leaf` would falsely return `Ok(Some(-inf_value))`.
    let absent_left = prover
        .perform_one_operation(&Operation::Lookup(key(0x05)))
        .expect("lookup in materialized left subtree should succeed");
    assert_eq!(
        absent_left, None,
        "absent key must not falsely match the -inf sentinel leaf"
    );

    // And an existing key in the left subtree must still resolve to its own value.
    let present_left = prover
        .perform_one_operation(&Operation::Lookup(key(0x10)))
        .expect("existing key in left subtree should succeed");
    assert_eq!(present_left, Some(val(0x10)));
}

/// Regression for `double_left_rotate` overwriting `this.tree.root` mid-traversal:
/// deleting `0x05` triggers `double_left_rotate` at depth 1 (inside `root.left`),
/// which reduces `root.left`'s height by 1; rebalancing at `root` (balance = +1)
/// then encounters an unresolvable `LabelOnly` right child and fails with `Err`.
/// On `Err`, `tree.root` must remain the original top root and persisted nodes
/// must stay unmodified.
#[test]
fn interior_double_left_rotation_does_not_overwrite_tree_root_on_ancestor_error() {
    let neg_inf = LeafNode::new(&key(0x00), &val(0x00), &key(0x05));
    let leaf_05 = LeafNode::new(&key(0x05), &val(0x05), &key(0x10));
    let leaf_10 = LeafNode::new(&key(0x10), &val(0x10), &key(0x20));
    let leaf_20 = LeafNode::new(&key(0x20), &val(0x20), &key(0x30));
    let leaf_30 = LeafNode::new(&key(0x30), &val(0x30), &key(0x50));

    let l_left = InternalNode::new(Some(key(0x05)), &neg_inf, &leaf_05, 0);
    let lr_left = InternalNode::new(Some(key(0x20)), &leaf_10, &leaf_20, 0);
    let l_right = InternalNode::new(Some(key(0x30)), &lr_left, &leaf_30, -1);
    let left_subtree = InternalNode::new(Some(key(0x10)), &l_left, &l_right, 1);

    let unloaded_right = Node::new_label(&[0xABu8; 32]);
    let top_root = InternalNode::new(Some(key(0x50)), &left_subtree, &unloaded_right, 1);

    let mut tree = AVLTree::new(unresolvable, 32, Some(8));
    tree.root = Some(top_root.clone());
    tree.height = 5;
    tree.reset();

    let mut prover = BatchAVLProver::new(tree, true);
    let before_digest = prover.digest().unwrap();

    let res = prover.perform_one_operation(&Operation::Remove(key(0x05)));
    assert!(
        res.is_err(),
        "rebalancing at root against unloaded right child must fail"
    );

    let after_root = prover.base.tree.root.as_ref().unwrap();
    assert!(
        Rc::ptr_eq(after_root, &top_root),
        "failed delete after interior double_left_rotate must not overwrite tree.root"
    );
    assert_eq!(prover.digest().unwrap(), before_digest);
    assert_eq!(prover.unauthenticated_lookup(&key(0x05)), Some(val(0x05)));
}

/// Symmetric regression for `double_right_rotate` overwriting `this.tree.root`:
/// deleting `0x90` triggers `double_right_rotate` at depth 1 (inside `root.right`),
/// which reduces `root.right`'s height by 1; rebalancing at `root` (balance = -1)
/// then encounters an unresolvable `LabelOnly` left child and fails with `Err`.
#[test]
fn interior_double_right_rotation_does_not_overwrite_tree_root_on_ancestor_error() {
    let leaf_50 = LeafNode::new(&key(0x50), &val(0x50), &key(0x60));
    let leaf_60 = LeafNode::new(&key(0x60), &val(0x60), &key(0x70));
    let leaf_70 = LeafNode::new(&key(0x70), &val(0x70), &key(0x80));
    let leaf_80 = LeafNode::new(&key(0x80), &val(0x80), &key(0x90));
    let leaf_90 = LeafNode::new(&key(0x90), &val(0x90), &key(0xFF));

    let rl_right = InternalNode::new(Some(key(0x70)), &leaf_60, &leaf_70, 0);
    let r_left = InternalNode::new(Some(key(0x60)), &leaf_50, &rl_right, 1);
    let r_right = InternalNode::new(Some(key(0x90)), &leaf_80, &leaf_90, 0);
    let right_subtree = InternalNode::new(Some(key(0x80)), &r_left, &r_right, -1);

    let unloaded_left = Node::new_label(&[0xCDu8; 32]);
    let top_root = InternalNode::new(Some(key(0x50)), &unloaded_left, &right_subtree, -1);

    let mut tree = AVLTree::new(unresolvable, 32, Some(8));
    tree.root = Some(top_root.clone());
    tree.height = 5;
    tree.reset();

    let mut prover = BatchAVLProver::new(tree, true);
    let before_digest = prover.digest().unwrap();

    let res = prover.perform_one_operation(&Operation::Remove(key(0x90)));
    assert!(
        res.is_err(),
        "rebalancing at root against unloaded left child must fail"
    );

    let after_root = prover.base.tree.root.as_ref().unwrap();
    assert!(
        Rc::ptr_eq(after_root, &top_root),
        "failed delete after interior double_right_rotate must not overwrite tree.root"
    );
    assert_eq!(prover.digest().unwrap(), before_digest);
    assert_eq!(prover.unauthenticated_lookup(&key(0x90)), Some(val(0x90)));
}

/// Regression for `AVLTree::extract_first_node`: once a matching leaf is found
/// in the left subtree, the right subtree must NOT be visited or resolved.
#[test]
fn extract_first_node_is_lazy_and_stops_after_left_match() {
    fn panic_resolver(_digest: &[u8; 32]) -> Node {
        panic!("right subtree resolver must not be invoked after left-subtree match");
    }

    let neg_inf = LeafNode::new(&key(0x00), &val(0x00), &key(0x10));
    let leaf_10 = LeafNode::new(&key(0x10), &val(0x10), &key(0x20));
    let left = InternalNode::new(Some(key(0x10)), &neg_inf, &leaf_10, 0);
    let unloaded_right = Node::new_label(&[0xEFu8; 32]);
    let root = InternalNode::new(Some(key(0x20)), &left, &unloaded_right, 0);

    let mut tree = AVLTree::new(panic_resolver, 32, Some(8));
    tree.root = Some(root);
    tree.height = 2;

    let visits = Cell::new(0usize);
    let found = tree.extract_first_node(&mut |node| {
        visits.set(visits.get() + 1);
        matches!(node, Node::Leaf(l) if l.hdr.key == Some(key(0x10)))
    });

    let matched = found.expect("should find key 0x10");
    assert_eq!(matched.borrow().key(), key(0x10));
    // Left subtree has 2 leaves (-inf, 0x10): -inf is visited first, then 0x10
    // matches on visit #2; the unloaded right child must not be visited or resolved.
    assert_eq!(
        visits.get(),
        2,
        "extract_first_node must short-circuit after matching in the left subtree"
    );
}
