use std::sync::Arc;

pub type BacklogReporter = Arc<dyn Fn(u64) + Send + Sync + 'static>;
pub type WriteReporter = Arc<dyn Fn(u64) + Send + Sync + 'static>;

#[derive(Clone, Default)]
pub struct ParallelFileWriterOptions {
    pub worker_threads: usize,
    pub max_backlog_bytes: usize,
    pub backlog_reporter: Option<BacklogReporter>,
    pub write_reporter: Option<WriteReporter>,
}
