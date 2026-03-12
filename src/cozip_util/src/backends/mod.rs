use std::fs::File;

use crate::error::ParallelFileReaderError;
use crate::error::ParallelFileWriterError;
use crate::file_reader::ParallelReadHandle;
use crate::options::ParallelFileReaderOptions;
use crate::options::ParallelFileWriterOptions;

#[cfg(not(any(target_os = "linux", windows)))]
mod fallback;
#[cfg(target_os = "linux")]
mod linux;
#[cfg(windows)]
mod windows;

pub(crate) trait FileWriterBackend: Send + Sync + 'static {
    fn submit(&self, offset: u64, data: Vec<u8>) -> Result<(), ParallelFileWriterError>;
    fn backlog_bytes(&self) -> u64;
    fn drain(&self) -> Result<(), ParallelFileWriterError>;
}

pub(crate) trait FileReaderBackend: Send + Sync + 'static {
    fn submit(&self, offset: u64, len: usize) -> Result<ParallelReadHandle, ParallelFileReaderError>;
    fn backlog_bytes(&self) -> u64;
    fn drain(&self) -> Result<(), ParallelFileReaderError>;
}

#[cfg(windows)]
pub(crate) fn create(
    file: File,
    options: ParallelFileWriterOptions,
) -> Result<Box<dyn FileWriterBackend>, ParallelFileWriterError> {
    Ok(Box::new(windows::WindowsFileWriter::new(file, options)?))
}

#[cfg(windows)]
pub(crate) fn create_reader(
    file: File,
    options: ParallelFileReaderOptions,
) -> Result<Box<dyn FileReaderBackend>, ParallelFileReaderError> {
    Ok(Box::new(windows::WindowsFileReader::new(file, options)?))
}

#[cfg(target_os = "linux")]
pub(crate) fn create(
    file: File,
    options: ParallelFileWriterOptions,
) -> Result<Box<dyn FileWriterBackend>, ParallelFileWriterError> {
    Ok(Box::new(linux::LinuxFileWriter::new(file, options)?))
}

#[cfg(target_os = "linux")]
pub(crate) fn create_reader(
    file: File,
    options: ParallelFileReaderOptions,
) -> Result<Box<dyn FileReaderBackend>, ParallelFileReaderError> {
    Ok(Box::new(linux::LinuxFileReader::new(file, options)?))
}

#[cfg(not(any(target_os = "linux", windows)))]
pub(crate) fn create(
    file: File,
    options: ParallelFileWriterOptions,
) -> Result<Box<dyn FileWriterBackend>, ParallelFileWriterError> {
    Ok(Box::new(fallback::FallbackFileWriter::new(file, options)?))
}

#[cfg(not(any(target_os = "linux", windows)))]
pub(crate) fn create_reader(
    file: File,
    options: ParallelFileReaderOptions,
) -> Result<Box<dyn FileReaderBackend>, ParallelFileReaderError> {
    Ok(Box::new(fallback::FallbackFileReader::new(file, options)?))
}
