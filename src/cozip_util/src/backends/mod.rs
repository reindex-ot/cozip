use std::fs::File;

use crate::error::ParallelFileWriterError;
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

#[cfg(windows)]
pub(crate) fn create(
    file: File,
    options: ParallelFileWriterOptions,
) -> Result<Box<dyn FileWriterBackend>, ParallelFileWriterError> {
    Ok(Box::new(windows::WindowsFileWriter::new(file, options)?))
}

#[cfg(target_os = "linux")]
pub(crate) fn create(
    file: File,
    options: ParallelFileWriterOptions,
) -> Result<Box<dyn FileWriterBackend>, ParallelFileWriterError> {
    Ok(Box::new(linux::LinuxFileWriter::new(file, options)?))
}

#[cfg(not(any(target_os = "linux", windows)))]
pub(crate) fn create(
    file: File,
    options: ParallelFileWriterOptions,
) -> Result<Box<dyn FileWriterBackend>, ParallelFileWriterError> {
    Ok(Box::new(fallback::FallbackFileWriter::new(file, options)?))
}
