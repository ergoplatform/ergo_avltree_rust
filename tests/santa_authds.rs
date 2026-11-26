//! Conformance against SANTA's `authds` vectors: AVL+ batch proofs and verifier results
//! recorded from the JVM reference (scrypto 3.0.0, `blessed_by: "jvm:scrypto-3.0.0"`).
//!
//! Vectors: `tests/santa_authds/*.json`, copied unchanged from
//! https://github.com/mwaddip/santa `vectors/authds/any/vendored/` at commit
//! 88856321d9e96b39292638b92402991da167152f (MIT). Format: SANTA's
//! `docs/contract/runner-contract-authds.md` (`santa-authds/v1`).

use bytes::Bytes;
use ergo_avltree_rust::authenticated_tree_ops::AuthenticatedTreeOps;
use ergo_avltree_rust::batch_avl_prover::BatchAVLProver;
use ergo_avltree_rust::batch_avl_verifier::BatchAVLVerifier;
use ergo_avltree_rust::batch_node::{AVLTree, Node, NodeHeader};
use ergo_avltree_rust::operation::{KeyDelta, KeyValue, Operation};
use serde_json::Value;

fn hex(s: &str) -> Bytes {
    Bytes::from(base16::decode(s.as_bytes()).expect("hex"))
}

fn to_hex(b: &[u8]) -> String {
    base16::encode_lower(b)
}

fn tree(settings: &Value) -> AVLTree {
    let key_length = settings["key_length"].as_u64().unwrap() as usize;
    let value_length = settings["value_length"].as_u64().map(|v| v as usize);
    AVLTree::new(
        |digest: &[u8; 32]| Node::LabelOnly(NodeHeader::new(Some(*digest), None)),
        key_length,
        value_length,
    )
}

fn op(o: &Value) -> Operation {
    let key = hex(o["key_hex"].as_str().unwrap());
    let kv = || KeyValue {
        key: key.clone(),
        value: hex(o["value_hex"].as_str().unwrap()),
    };
    match o["tag"].as_str().unwrap() {
        "Insert" => Operation::Insert(kv()),
        "Update" => Operation::Update(kv()),
        "InsertOrUpdate" => Operation::InsertOrUpdate(kv()),
        "Remove" => Operation::Remove(key),
        "RemoveIfExists" => Operation::RemoveIfExists(key),
        "Lookup" => Operation::Lookup(key),
        "UpdateLongBy" => Operation::UpdateLongBy(KeyDelta {
            key,
            delta: o["delta"].as_str().unwrap().parse().unwrap(),
        }),
        t => panic!("unsupported tag {t}"),
    }
}

fn load(name: &str) -> Vec<Value> {
    let path = format!("{}/tests/santa_authds/{name}", env!("CARGO_MANIFEST_DIR"));
    let doc: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    assert_eq!(doc["schema"], "santa-authds/v1");
    assert_eq!(doc["blessed_by"], "jvm:scrypto-3.0.0");
    doc["entries"].as_array().unwrap().clone()
}

/// One entry's disagreements with the JVM, empty when it conforms.
fn prove_mismatches(e: &Value) -> Vec<String> {
    let collect_changed_nodes = false;
    let mut prover = BatchAVLProver::new(tree(&e["settings"]), collect_changed_nodes);
    let after: Vec<u64> = e["payload"]["gen_proof_after"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    let (mut proofs, mut digests) = (vec![], vec![]);
    for (i, o) in e["payload"]["operations"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        if let Err(err) = prover.perform_one_operation(&op(o)) {
            return vec![format!("operations[{i}]: prover error: {err}")];
        }
        if after.contains(&(i as u64)) {
            proofs.push(to_hex(&prover.generate_proof()));
            match prover.digest() {
                Some(d) => digests.push(to_hex(&d)),
                None => return vec![format!("operations[{i}]: no digest after the proof")],
            }
        }
    }
    let mut out = vec![];
    for (k, (got, want)) in [("proofs", &proofs), ("digests", &digests)]
        .iter()
        .map(|(k, v)| (*k, (*v, &e["expected"][*k])))
    {
        let want: Vec<String> = want
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        if got != &want {
            out.push(format!("{k}: got {got:?}, want {want:?}"));
        }
    }
    out
}

fn verify_mismatches(e: &Value) -> Vec<String> {
    let s = &e["settings"];
    let p = &e["payload"];
    let x = &e["expected"];
    let verifier = BatchAVLVerifier::new(
        &hex(p["starting_digest_hex"].as_str().unwrap()),
        &hex(p["proof_hex"].as_str().unwrap()),
        tree(s),
        s["max_num_operations"].as_u64().map(|v| v as usize),
        s["max_deletes"].as_u64().map(|v| v as usize),
    );
    let mut verifier = match verifier {
        Ok(v) if v.digest().is_some() => Some(v),
        _ => None,
    };
    let accepted = verifier.is_some();
    let mut out = vec![];
    if accepted != x["proof_accepted"].as_bool().unwrap() {
        out.push(format!("proof_accepted: got {accepted}"));
        return out;
    }
    let Some(v) = verifier.as_mut() else {
        return out;
    };
    let (ops, results) = (
        p["operations"].as_array().unwrap(),
        x["results"].as_array().unwrap(),
    );
    if ops.len() != results.len() {
        out.push(format!(
            "{} operations, {} expected results",
            ops.len(),
            results.len()
        ));
        return out;
    }
    for (i, (o, want)) in ops.iter().zip(results).enumerate() {
        let got = v.perform_one_operation(&op(o));
        let (ok, value) = match &got {
            Ok(Some(b)) => (true, Some(to_hex(b))),
            Ok(None) => (true, None),
            Err(_) => (false, None),
        };
        let want_value = want["value"].as_str().map(|s| s.to_string());
        if ok != want["ok"].as_bool().unwrap() || value != want_value {
            out.push(format!(
                "results[{i}]: got ok={ok} value={value:?}, want {want}"
            ));
        }
    }
    let digest = v.digest().map(|d| to_hex(&d));
    let want_digest = x["new_digest_hex"].as_str().map(|s| s.to_string());
    if digest != want_digest {
        out.push(format!("new_digest: got {digest:?}, want {want_digest:?}"));
    }
    out
}

fn run(name: &str, check: fn(&Value) -> Vec<String>) {
    let entries = load(name);
    let mut failed = vec![];
    for e in &entries {
        let m = check(e);
        if !m.is_empty() {
            failed.push(format!("{}: {}", e["name"].as_str().unwrap(), m.join("; ")));
        }
    }
    println!(
        "{name}: {} of {} entries agree with the JVM",
        entries.len() - failed.len(),
        entries.len()
    );
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

#[test]
fn prover_matches_the_jvm_on_santa_avl_prove() {
    run("AvlProve.ergots_corpus.json", prove_mismatches);
}

#[test]
fn verifier_matches_the_jvm_on_santa_avl_verify() {
    run("AvlVerify.ergots_corpus.json", verify_mismatches);
}
