use bytes::Bytes;
use ergo_avltree_rust::batch_node::{AVLTree, LeafNode, Node, NodeHeader, NodeId};
use ergo_avltree_rust::operation::Digest32;

fn label_resolver(digest: &Digest32) -> Node {
    Node::LabelOnly(NodeHeader::new(Some(*digest), None))
}

fn leaf(fill: u8) -> NodeId {
    let key = Bytes::from(vec![fill; 32]);
    let value = Bytes::from(vec![fill; 8]);
    let next_key = Bytes::from(vec![fill.wrapping_add(1); 32]);
    LeafNode::new(&key, &value, &next_key)
}

#[test]
fn contains_is_conservative_for_unresolved_label_only_root() {
    let mut tree = AVLTree::new(label_resolver, 32, None);
    tree.root = Some(Node::new_label(&[0xA5; 32]));

    assert!(
        tree.contains(&leaf(0x11)),
        "an unresolved subtree is only maybe absent and must not authorize deletion"
    );
}

#[test]
fn contains_rejects_a_nonmatching_resolved_leaf() {
    let mut tree = AVLTree::new(label_resolver, 32, None);
    tree.root = Some(leaf(0x22));

    assert!(!tree.contains(&leaf(0x33)));
}
