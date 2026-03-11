use std::collections::VecDeque;
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use super::FileWriterBackend;
use crate::error::ParallelFileWriterError;
use crate::options::{BacklogReporter, ParallelFileWriterOptions, WriteReporter};

struct WriteTask {
    offset: u64,
    data: Vec<u8>,
}

#[derive(Default)]
struct SharedState {
    queue: VecDeque<WriteTask>,
    pending_bytes: usize,
    active_workers: usize,
    closed: bool,
    stopped: bool,
}

pub(crate) struct FallbackFileWriter {
    state: Arc<(Mutex<SharedState>, Condvar)>,
    error: Arc<Mutex<Option<ParallelFileWriterError>>>,
    backlog_bytes: Arc<AtomicU64>,
    max_backlog_bytes: usize,
    backlog_reporter: Option<BacklogReporter>,
}

impl FallbackFileWriter {
    pub(crate) fn new(
        file: File,
        options: ParallelFileWriterOptions,
    ) -> Result<Self, ParallelFileWriterError> {
        let workers = options.worker_threads.max(1);
        let state = Arc::new((
            Mutex::new(SharedState {
                active_workers: workers,
                ..SharedState::default()
            }),
            Condvar::new(),
        ));
        let error = Arc::new(Mutex::new(None));
        let backlog_bytes = Arc::new(AtomicU64::new(0));
        let file = Arc::new(Mutex::new(file));
        for _ in 0..workers {
            let state_ref = Arc::clone(&state);
            let error_ref = Arc::clone(&error);
            let backlog_ref = Arc::clone(&backlog_bytes);
            let file_ref = Arc::clone(&file);
            let backlog_reporter = options.backlog_reporter.clone();
            let write_reporter = options.write_reporter.clone();
            thread::spawn(move || {
                worker_loop(
                    file_ref,
                    state_ref,
                    error_ref,
                    backlog_ref,
                    backlog_reporter,
                    write_reporter,
                );
            });
        }
        Ok(Self {
            state,
            error,
            backlog_bytes,
            max_backlog_bytes: options.max_backlog_bytes.max(1),
            backlog_reporter: options.backlog_reporter,
        })
    }

    fn check_error(&self) -> Result<(), ParallelFileWriterError> {
        let mut slot = self
            .error
            .lock()
            .map_err(|_| std::io::Error::other("fallback writer error slot poisoned"))?;
        if let Some(error) = slot.take() {
            return Err(error);
        }
        Ok(())
    }

    fn update_backlog(&self, bytes: usize) {
        let bytes64 = u64::try_from(bytes).unwrap_or(u64::MAX);
        self.backlog_bytes.store(bytes64, Ordering::Relaxed);
        if let Some(reporter) = &self.backlog_reporter {
            reporter(bytes64);
        }
    }
}

impl FileWriterBackend for FallbackFileWriter {
    fn submit(&self, offset: u64, data: Vec<u8>) -> Result<(), ParallelFileWriterError> {
        self.check_error()?;
        let (lock, cv) = &*self.state;
        let mut state = lock
            .lock()
            .map_err(|_| std::io::Error::other("fallback writer state poisoned"))?;
        while state.pending_bytes >= self.max_backlog_bytes && !state.stopped && !state.closed {
            state = cv
                .wait(state)
                .map_err(|_| std::io::Error::other("fallback writer state poisoned"))?;
            self.check_error()?;
        }
        if state.stopped || state.closed {
            self.check_error()?;
            return Err(ParallelFileWriterError::Closed);
        }
        state.pending_bytes = state.pending_bytes.saturating_add(data.len());
        state.queue.push_back(WriteTask { offset, data });
        self.update_backlog(state.pending_bytes);
        cv.notify_all();
        Ok(())
    }

    fn backlog_bytes(&self) -> u64 {
        self.backlog_bytes.load(Ordering::Relaxed)
    }

    fn drain(&self) -> Result<(), ParallelFileWriterError> {
        let (lock, cv) = &*self.state;
        let mut state = lock
            .lock()
            .map_err(|_| std::io::Error::other("fallback writer state poisoned"))?;
        state.closed = true;
        cv.notify_all();
        while !(state.queue.is_empty() && state.active_workers == 0) && !state.stopped {
            state = cv
                .wait(state)
                .map_err(|_| std::io::Error::other("fallback writer state poisoned"))?;
        }
        drop(state);
        self.check_error()
    }
}

fn worker_loop(
    file_ref: Arc<Mutex<File>>,
    state_ref: Arc<(Mutex<SharedState>, Condvar)>,
    error_ref: Arc<Mutex<Option<ParallelFileWriterError>>>,
    backlog_bytes: Arc<AtomicU64>,
    backlog_reporter: Option<BacklogReporter>,
    write_reporter: Option<WriteReporter>,
) {
    loop {
        let task = {
            let (lock, cv) = &*state_ref;
            let mut state = match lock.lock() {
                Ok(guard) => guard,
                Err(_) => return,
            };
            loop {
                if let Some(task) = state.queue.pop_front() {
                    break Some(task);
                }
                if state.closed || state.stopped {
                    state.active_workers = state.active_workers.saturating_sub(1);
                    cv.notify_all();
                    break None;
                }
                state = match cv.wait(state) {
                    Ok(guard) => guard,
                    Err(_) => return,
                };
            }
        };
        let Some(task) = task else {
            return;
        };
        let write_result = (|| -> Result<(), ParallelFileWriterError> {
            let mut file = file_ref
                .lock()
                .map_err(|_| std::io::Error::other("fallback file mutex poisoned"))?;
            file.seek(SeekFrom::Start(task.offset))?;
            file.write_all(&task.data)?;
            Ok(())
        })();
        let (lock, cv) = &*state_ref;
        if let Ok(mut state) = lock.lock() {
            state.pending_bytes = state.pending_bytes.saturating_sub(task.data.len());
            let backlog = u64::try_from(state.pending_bytes).unwrap_or(u64::MAX);
            backlog_bytes.store(backlog, Ordering::Relaxed);
            if let Some(reporter) = &backlog_reporter {
                reporter(backlog);
            }
            match write_result {
                Ok(()) => {
                    if let Some(reporter) = &write_reporter {
                        reporter(u64::try_from(task.data.len()).unwrap_or(u64::MAX));
                    }
                }
                Err(error) => {
                    state.stopped = true;
                    if let Ok(mut slot) = error_ref.lock() {
                        if slot.is_none() {
                            *slot = Some(error);
                        }
                    }
                }
            }
            cv.notify_all();
        }
    }
}
