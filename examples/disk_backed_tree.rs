//! Basic example demonstrating disk-backed AVL tree usage
//!
//! This example shows:
//! - Creating a disk-backed AVL tree
//! - Performing insert, update, lookup operations
//! - Committing changes to disk
//! - Rollback to previous versions
//! - Reopening the database

use bytes::Bytes;
use ergo_avltree_rust::batch_avl_prover::BatchAVLProver;
use ergo_avltree_rust::batch_node::*;
use ergo_avltree_rust::operation::*;
use ergo_avltree_rust::persistent_batch_avl_prover::*;
use ergo_avltree_rust::storage::DiskBackedAVLStorage;

fn main() -> anyhow::Result<()> {
    println!("=== Disk-Backed AVL Tree Example ===\n");

    // Configuration
    let db_path = "./example_db";
    let key_length = 32;
    let value_length = Some(8); // Fixed 8-byte values
    let keep_versions = 5; // Keep last 5 versions for rollback

    // Clean up old database if exists
    if std::path::Path::new(db_path).exists() {
        std::fs::remove_dir_all(db_path)?;
    }

    // Create disk-backed storage
    println!("Creating disk-backed storage at: {}", db_path);
    let storage = Box::new(DiskBackedAVLStorage::new(
        db_path,
        key_length,
        value_length,
        keep_versions,
    )?);

    // Create resolver for lazy loading
    let resolver = DiskBackedAVLStorage::get_resolver();

    // Create AVL tree with disk resolver
    let tree = AVLTree::new(resolver, key_length, value_length);
    let prover = BatchAVLProver::new(tree, true);

    // Create persistent prover
    let mut persistent_prover = PersistentBatchAVLProver::new(prover, storage, vec![])?;

    println!("Initial digest: {}\n", hex::encode(&persistent_prover.digest()));

    // Example 1: Insert operations
    println!("--- Example 1: Inserting data ---");
    for i in 1..=5 {
        let mut key = vec![0u8; key_length];
        key[31] = i;
        let value = Bytes::from((i as u64 * 100).to_be_bytes().to_vec());

        let op = Operation::Insert(KeyValue {
            key: Bytes::from(key.clone()),
            value: value.clone(),
        });

        persistent_prover.perform_one_operation(&op)?;
        println!("Inserted key {:?} with value: {}", &key[30..], i as u64 * 100);
    }

    // Commit to disk
    persistent_prover.generate_proof_and_update_storage(vec![])?;
    let digest_after_insert = persistent_prover.digest();
    println!("Committed to disk. New digest: {}\n", hex::encode(&digest_after_insert));

    // Example 2: Lookup operations
    println!("--- Example 2: Looking up values ---");
    for i in 1..=5 {
        let mut key = vec![0u8; key_length];
        key[31] = i;

        if let Some(value) = persistent_prover.unauthenticated_lookup(&Bytes::from(key.clone())) {
            let val = u64::from_be_bytes(value.to_vec().try_into().unwrap());
            println!("Key {:?} -> Value: {}", &key[30..], val);
        }
    }
    println!();

    // Example 3: Update operations
    println!("--- Example 3: Updating values ---");
    let mut key = vec![0u8; key_length];
    key[31] = 3;
    let new_value = Bytes::from(999u64.to_be_bytes().to_vec());

    let op = Operation::Update(KeyValue {
        key: Bytes::from(key.clone()),
        value: new_value.clone(),
    });
    persistent_prover.perform_one_operation(&op)?;
    persistent_prover.generate_proof_and_update_storage(vec![])?;
    let digest_after_update = persistent_prover.digest();

    println!("Updated key {:?} to value: 999", &key[30..]);
    println!("New digest: {}\n", hex::encode(&digest_after_update));

    // Example 4: Rollback
    println!("--- Example 4: Rollback to previous version ---");
    println!("Rolling back to digest: {}", hex::encode(&digest_after_insert));
    persistent_prover.rollback(&digest_after_insert)?;

    // Verify rollback worked
    if let Some(value) = persistent_prover.unauthenticated_lookup(&Bytes::from(key.clone())) {
        let val = u64::from_be_bytes(value.to_vec().try_into().unwrap());
        println!("After rollback, key {:?} -> Value: {} (should be 300)", &key[30..], val);
    }
    println!();

    // Example 5: Remove operation
    println!("--- Example 5: Removing keys ---");
    key[31] = 2;
    let op = Operation::Remove(Bytes::from(key.clone()));
    persistent_prover.perform_one_operation(&op)?;
    persistent_prover.generate_proof_and_update_storage(vec![])?;

    println!("Removed key {:?}", &key[30..]);
    match persistent_prover.unauthenticated_lookup(&Bytes::from(key.clone())) {
        None => println!("Lookup after removal: None (as expected)"),
        Some(_) => println!("ERROR: Key still exists!"),
    }
    println!();

    // Example 6: UpdateLongBy operation
    println!("--- Example 6: UpdateLongBy (increment/decrement) ---");
    key[31] = 10;

    // Increment non-existing key
    let op = Operation::UpdateLongBy(KeyDelta {
        key: Bytes::from(key.clone()),
        delta: 50,
    });
    persistent_prover.perform_one_operation(&op)?;
    println!("Created key {:?} with delta: +50", &key[30..]);

    // Increment again
    let op = Operation::UpdateLongBy(KeyDelta {
        key: Bytes::from(key.clone()),
        delta: 25,
    });
    persistent_prover.perform_one_operation(&op)?;
    println!("Incremented key {:?} with delta: +25", &key[30..]);

    persistent_prover.generate_proof_and_update_storage(vec![])?;

    if let Some(value) = persistent_prover.unauthenticated_lookup(&Bytes::from(key.clone())) {
        let val = i64::from_be_bytes(value.to_vec().try_into().unwrap());
        println!("Current value of key {:?}: {} (should be 75)", &key[30..], val);
    }
    println!();

    println!("--- Final Statistics ---");
    println!("Tree height: {}", persistent_prover.height());
    println!("Final digest: {}", hex::encode(&persistent_prover.digest()));
    println!("Available versions: {}", persistent_prover.storage.rollback_versions().count());

    // Cleanup
    println!("\nCleaning up example database...");
    drop(persistent_prover);
    std::fs::remove_dir_all(db_path)?;

    println!("Example completed successfully!");
    Ok(())
}
