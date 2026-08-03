use blake2::digest::Digest;
use bytes::Bytes;
use ergo_avltree_rust::batch_node::{Blake2b256, Node, NodeId, SerializedAdProof};
use ergo_avltree_rust::operation::{ADDigest, Digest32};

const LEAF: u8 = 2;
const LABEL: u8 = 3;
const END_OF_TREE: u8 = 4;

pub const DEEP_DROP_DEPTH: usize = 50_000;
pub const LOOKUP_KEY: [u8; 1] = [0x10];
pub const LEAF_VALUE: [u8; 1] = [0xaa];

pub fn deep_spine_proof(depth: usize, lookup_directions: bool) -> SerializedAdProof {
    let mut proof = Vec::new();
    proof.extend_from_slice(&[LEAF, LOOKUP_KEY[0], 0x20, LEAF_VALUE[0]]);
    for _ in 0..depth {
        proof.push(LABEL);
        proof.extend_from_slice(&[0x11; 32]);
        proof.push(0x00);
    }
    proof.push(END_OF_TREE);
    if lookup_directions {
        proof.extend(core::iter::repeat(0xff).take((depth + 7) / 8));
    } else {
        proof.push(0xff);
    }
    Bytes::from(proof)
}

pub fn deep_spine_root_label(depth: usize) -> Digest32 {
    let mut hasher = Blake2b256::new();
    hasher.update([0]);
    hasher.update(LOOKUP_KEY);
    hasher.update(LEAF_VALUE);
    hasher.update([0x20]);
    let mut label = Digest32::default();
    label.copy_from_slice(&hasher.finalize());

    for _ in 0..depth {
        let mut hasher = Blake2b256::new();
        hasher.update([1]);
        hasher.update([0]);
        hasher.update(label);
        hasher.update([0x11; 32]);
        label.copy_from_slice(&hasher.finalize());
    }
    label
}

pub fn hash_matching_digest(depth: usize, digest_height_byte: u8) -> ADDigest {
    let mut digest = deep_spine_root_label(depth).to_vec();
    digest.push(digest_height_byte);
    Bytes::from(digest)
}

pub fn assert_left_spine_depth(root: &NodeId, expected: usize) {
    let mut current = root.clone();
    for _ in 0..expected {
        let next = {
            let node = current.borrow();
            match &*node {
                Node::Internal(node) => node.left.clone(),
                _ => panic!("left spine ended before expected depth"),
            }
        };
        current = next;
    }
    assert!(matches!(&*current.borrow(), Node::Leaf(_)));
}
