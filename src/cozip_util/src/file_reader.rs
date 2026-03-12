use std::fs::File;
use std::sync::mpsc::Receiver;

use crate::backends::{self, FileReaderBackend};
use crate::error::ParallelFileReaderError;
use crate::options::ParallelFileReaderOptions;

pub struct ParallelReadHandle {
    receiver: Receiver<Result<Vec<u8>, ParallelFileReaderError>>,
}

impl ParallelReadHandle {
    pub(crate) fn new(
        receiver: Receiver<Result<Vec<u8>, ParallelFileReaderError>>,
    ) -> Self {
        Self { receiver }
    }

    pub fn recv(self) -> Result<Vec<u8>, ParallelFileReaderError> {
        self.receiver
            .recv()
            .map_err(|_| ParallelFileReaderError::ResponseClosed)?
    }
}

pub struct ParallelFileReader {
    backend: Box<dyn FileReaderBackend>,
}

impl ParallelFileReader {
    pub fn new(
        file: File,
        options: ParallelFileReaderOptions,
    ) -> Result<Self, ParallelFileReaderError> {
        Ok(Self {
            backend: backends::create_reader(file, options)?,
        })
    }

    pub fn submit(
        &self,
        offset: u64,
        len: usize,
    ) -> Result<ParallelReadHandle, ParallelFileReaderError> {
        self.backend.submit(offset, len)
    }

    pub fn backlog_bytes(&self) -> u64 {
        self.backend.backlog_bytes()
    }

    pub fn drain(&self) -> Result<(), ParallelFileReaderError> {
        self.backend.drain()
    }
}
