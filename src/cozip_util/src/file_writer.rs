use std::fs::File;

use crate::backends::{self, FileWriterBackend};
use crate::error::ParallelFileWriterError;
use crate::options::ParallelFileWriterOptions;

pub struct ParallelFileWriter {
    backend: Box<dyn FileWriterBackend>,
}

impl ParallelFileWriter {
    pub fn new(
        file: File,
        options: ParallelFileWriterOptions,
    ) -> Result<Self, ParallelFileWriterError> {
        Ok(Self {
            backend: backends::create(file, options)?,
        })
    }

    pub fn submit(&self, offset: u64, data: Vec<u8>) -> Result<(), ParallelFileWriterError> {
        self.backend.submit(offset, data)
    }

    pub fn backlog_bytes(&self) -> u64 {
        self.backend.backlog_bytes()
    }

    pub fn drain(&self) -> Result<(), ParallelFileWriterError> {
        self.backend.drain()
    }
}
