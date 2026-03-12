mod backends;
mod error;
mod file_reader;
mod file_writer;
mod options;

pub use error::ParallelFileReaderError;
pub use error::ParallelFileWriterError;
pub use file_reader::{ParallelFileReader, ParallelReadHandle};
pub use file_writer::ParallelFileWriter;
pub use options::{
    BacklogReporter, ParallelFileReaderOptions, ParallelFileWriterOptions, ReadReporter,
    WriteReporter,
};
