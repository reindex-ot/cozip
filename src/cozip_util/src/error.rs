use thiserror::Error;

#[derive(Debug, Error)]
pub enum ParallelFileWriterError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("parallel file writer queue is closed")]
    Closed,
    #[error("parallel file writer backlog exceeded usize range")]
    NumericOverflow,
}
