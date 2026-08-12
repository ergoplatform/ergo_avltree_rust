use crate::authenticated_tree_ops::*;
use crate::batch_avl_prover::*;
use crate::batch_node::*;
use crate::operation::*;
use crate::versioned_avl_storage::*;
use alloc::boxed::Box;
use alloc::vec::Vec;
use anyhow::{ensure, Result};

pub struct PersistentBatchAVLProver {
    pub prover: BatchAVLProver,
    pub storage: Box<dyn VersionedAVLStorage>,
}

impl PersistentBatchAVLProver {
    pub fn new(
        prover: BatchAVLProver,
        storage: Box<dyn VersionedAVLStorage>,
        additional_data: Vec<(ADKey, ADValue)>,
    ) -> Result<PersistentBatchAVLProver> {
        let mut this = PersistentBatchAVLProver { prover, storage };
        match this.storage.version() {
            Some(ver) => {
                let _ = this.rollback(&ver)?;
            }
            None => {
                let _ = this.generate_proof_and_update_storage(additional_data)?;
            }
        }
        ensure!(this.storage.version().unwrap() == this.digest());
        Ok(this)
    }

    pub fn digest(&self) -> ADDigest {
        self.prover.digest().unwrap()
    }

    pub fn height(&self) -> usize {
        self.prover.base.tree.height
    }

    pub fn prover<'a>(&'a mut self) -> &'a mut BatchAVLProver {
        &mut self.prover
    }

    pub fn unauthenticated_lookup(&self, key: &ADKey) -> Option<ADValue> {
        self.prover.unauthenticated_lookup(key)
    }

    pub fn perform_one_operation(&mut self, operation: &Operation) -> Result<Option<ADValue>> {
        self.prover.perform_one_operation(operation)
    }

    pub fn generate_proof_and_update_storage(
        &mut self,
        additional_data: Vec<(ADKey, ADValue)>,
    ) -> Result<SerializedAdProof> {
        self.storage.update(&mut self.prover, additional_data)?;
        Ok(self.prover.generate_proof())
    }

    /// Rewind to a stored version, abandoning any in-flight proof cycle.
    ///
    /// Delegates to [`BatchAVLProver::restore_root`] rather than re-installing
    /// the root by hand: both are the same rewind, and the hand-rolled copy
    /// kept drifting behind it — first missing the `old_top_node` rebase, then
    /// the `modified_nodes` clear. One implementation cannot drift from itself.
    pub fn rollback(&mut self, version: &ADDigest) -> Result<()> {
        let (root, height) = self.storage.rollback(version)?;
        self.prover.restore_root(root, height);
        Ok(())
    }
}
