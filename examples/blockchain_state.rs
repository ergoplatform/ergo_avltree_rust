//! Blockchain state management example using disk-backed AVL tree
//!
//! This example demonstrates:
//! - Managing blockchain state (account balances)
//! - Creating versioned snapshots per block
//! - Handling transactions with proof generation
//! - Rollback capability for chain reorganization
//! - Database persistence and recovery

use bytes::Bytes;
use ergo_avltree_rust::authenticated_tree_ops::*;
use ergo_avltree_rust::batch_avl_prover::BatchAVLProver;
use ergo_avltree_rust::batch_node::*;
use ergo_avltree_rust::operation::*;
use ergo_avltree_rust::persistent_batch_avl_prover::*;
use ergo_avltree_rust::storage::DiskBackedAVLStorage;
use std::collections::HashMap;

// Simple transaction struct
#[derive(Debug, Clone)]
struct Transaction {
    from: Vec<u8>,
    to: Vec<u8>,
    amount: u64,
}

// Block containing transactions
#[derive(Debug)]
struct Block {
    height: u32,
    transactions: Vec<Transaction>,
    state_digest: Option<Bytes>,
}

fn main() -> anyhow::Result<()> {
    println!("=== Blockchain State Management Example ===\n");

    let db_path = "./blockchain_state_db";
    let key_length = 32; // Account addresses (e.g., public key hashes)
    let value_length = Some(8); // Balance as u64
    let keep_versions = 100; // Keep last 100 block states

    // Clean up old database
    if std::path::Path::new(db_path).exists() {
        std::fs::remove_dir_all(db_path)?;
    }

    // Initialize storage
    println!("Initializing blockchain state database...");
    let storage = Box::new(DiskBackedAVLStorage::new(
        db_path,
        key_length,
        value_length,
        keep_versions,
    )?);

    let resolver = DiskBackedAVLStorage::get_resolver();
    let tree = AVLTree::new(resolver, key_length, value_length);
    let prover = BatchAVLProver::new(tree, true);
    let mut state = PersistentBatchAVLProver::new(prover, storage, vec![])?;

    println!("Genesis state digest: {}\n", hex::encode(&state.digest()));

    // Track block digests for rollback capability
    let mut block_digests: HashMap<u32, Bytes> = HashMap::new();
    block_digests.insert(0, state.digest());

    // Create some accounts with initial balances
    println!("--- Genesis Block: Creating initial accounts ---");
    let accounts = vec![
        (b"Alice___________________________".to_vec(), 1000u64),
        (b"Bob_____________________________".to_vec(), 2000u64),
        (b"Charlie_________________________".to_vec(), 1500u64),
        (b"Dave____________________________".to_vec(), 500u64),
    ];

    for (account, balance) in &accounts {
        let op = Operation::Insert(KeyValue {
            key: Bytes::from(account.clone()),
            value: Bytes::from(balance.to_be_bytes().to_vec()),
        });
        state.perform_one_operation(&op)?;
        println!("  Created account {} with balance: {}", 
                 String::from_utf8_lossy(&account[..7]), balance);
    }

    state.generate_proof_and_update_storage(vec![])?;
    let genesis_digest = state.digest();
    block_digests.insert(0, genesis_digest.clone());
    println!("Genesis block committed. Digest: {}\n", hex::encode(&genesis_digest));

    // Block 1: Process transactions
    println!("--- Block 1: Processing transactions ---");
    let block1_txs = vec![
        Transaction {
            from: accounts[0].0.clone(), // Alice
            to: accounts[1].0.clone(),   // Bob
            amount: 100,
        },
        Transaction {
            from: accounts[1].0.clone(), // Bob
            to: accounts[2].0.clone(),   // Charlie
            amount: 200,
        },
    ];

    process_block(&mut state, 1, block1_txs, &mut block_digests)?;

    // Block 2: More transactions
    println!("--- Block 2: Processing transactions ---");
    let block2_txs = vec![
        Transaction {
            from: accounts[2].0.clone(), // Charlie
            to: accounts[3].0.clone(),   // Dave
            amount: 300,
        },
        Transaction {
            from: accounts[0].0.clone(), // Alice
            to: accounts[3].0.clone(),   // Dave
            amount: 150,
        },
    ];

    process_block(&mut state, 2, block2_txs, &mut block_digests)?;

    // Display current balances
    println!("--- Current Account Balances ---");
    for (account, _) in &accounts {
        if let Some(balance_bytes) = state.unauthenticated_lookup(&Bytes::from(account.clone())) {
            let balance = u64::from_be_bytes(balance_bytes.to_vec().try_into().unwrap());
            println!("  {}: {}", String::from_utf8_lossy(&account[..7]), balance);
        }
    }
    println!();

    // Simulate chain reorganization - rollback to block 1
    println!("--- Simulating Chain Reorganization ---");
    println!("Rolling back from block 2 to block 1...");
    let block1_digest = block_digests.get(&1).unwrap();
    state.rollback(block1_digest)?;

    println!("Rollback successful. Current state digest: {}", hex::encode(&state.digest()));
    println!("\n--- Account Balances After Rollback ---");
    for (account, _) in &accounts {
        if let Some(balance_bytes) = state.unauthenticated_lookup(&Bytes::from(account.clone())) {
            let balance = u64::from_be_bytes(balance_bytes.to_vec().try_into().unwrap());
            println!("  {}: {}", String::from_utf8_lossy(&account[..7]), balance);
        }
    }
    println!();

    // Alternative block 2' (different from original block 2)
    println!("--- Block 2' (Alternative): Processing different transactions ---");
    let block2_prime_txs = vec![
        Transaction {
            from: accounts[1].0.clone(), // Bob
            to: accounts[0].0.clone(),   // Alice
            amount: 400,
        },
    ];

    process_block(&mut state, 2, block2_prime_txs, &mut block_digests)?;

    println!("--- Final Account Balances (Alternative Chain) ---");
    for (account, _) in &accounts {
        if let Some(balance_bytes) = state.unauthenticated_lookup(&Bytes::from(account.clone())) {
            let balance = u64::from_be_bytes(balance_bytes.to_vec().try_into().unwrap());
            println!("  {}: {}", String::from_utf8_lossy(&account[..7]), balance);
        }
    }
    println!();

    // Demonstrate database persistence
    println!("--- Testing Database Persistence ---");
    let final_digest = state.digest();
    println!("Current digest: {}", hex::encode(&final_digest));
    
    // Drop the state to close database
    drop(state);
    println!("Closed database connection.");

    // Reopen database
    println!("Reopening database...");
    let storage = Box::new(DiskBackedAVLStorage::new(
        db_path,
        key_length,
        value_length,
        keep_versions,
    )?);
    let resolver = DiskBackedAVLStorage::get_resolver();
    let tree = AVLTree::new(resolver, key_length, value_length);
    let prover = BatchAVLProver::new(tree, true);
    let state = PersistentBatchAVLProver::new(prover, storage, vec![])?;

    println!("Reopened digest: {}", hex::encode(&state.digest()));
    println!("Digests match: {}", state.digest() == final_digest);
    println!("Available rollback versions: {}", state.storage.rollback_versions().count());

    // Verify data is still accessible
    println!("\n--- Verifying Data After Reopen ---");
    for (account, _) in &accounts {
        if let Some(balance_bytes) = state.unauthenticated_lookup(&Bytes::from(account.clone())) {
            let balance = u64::from_be_bytes(balance_bytes.to_vec().try_into().unwrap());
            println!("  {}: {}", String::from_utf8_lossy(&account[..7]), balance);
        }
    }

    // Cleanup
    println!("\nCleaning up database...");
    drop(state);
    std::fs::remove_dir_all(db_path)?;

    println!("\n=== Example completed successfully! ===");
    Ok(())
}

fn process_block(
    state: &mut PersistentBatchAVLProver,
    block_height: u32,
    transactions: Vec<Transaction>,
    block_digests: &mut HashMap<u32, Bytes>,
) -> anyhow::Result<()> {
    println!("Processing block {} with {} transactions", block_height, transactions.len());

    for (i, tx) in transactions.iter().enumerate() {
        println!("  Tx {}: {} -> {} (amount: {})",
                 i + 1,
                 String::from_utf8_lossy(&tx.from[..7]),
                 String::from_utf8_lossy(&tx.to[..7]),
                 tx.amount);

        // Deduct from sender
        let from_key = Bytes::from(tx.from.clone());
        let op_deduct = Operation::UpdateLongBy(KeyDelta {
            key: from_key,
            delta: -(tx.amount as i64),
        });
        state.perform_one_operation(&op_deduct)?;

        // Add to receiver
        let to_key = Bytes::from(tx.to.clone());
        let op_add = Operation::UpdateLongBy(KeyDelta {
            key: to_key,
            delta: tx.amount as i64,
        });
        state.perform_one_operation(&op_add)?;
    }

    // Generate proof and commit state
    let _proof = state.generate_proof_and_update_storage(vec![])?;
    let digest = state.digest();
    block_digests.insert(block_height, digest.clone());

    println!("  Block {} committed. State digest: {}\n", block_height, hex::encode(&digest));
    Ok(())
}
