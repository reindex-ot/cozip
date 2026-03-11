use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io;
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

use io_uring::{IoUring, opcode, types};

use super::FileWriterBackend;
use crate::error::ParallelFileWriterError;
use crate::options::{BacklogReporter, ParallelFileWriterOptions, WriteReporter};

struct WriteRequest {
    id: u64,
    offset: u64,
    data: Vec<u8>,
}

struct SharedState {
    pending_bytes: usize,
    inflight_ops: usize,
    closed: bool,
    stopped: bool,
}

pub(crate) struct LinuxFileWriter {
    state: Arc<(Mutex<SharedState>, Condvar)>,
    error: Arc<Mutex<Option<ParallelFileWriterError>>>,
    backlog_bytes: Arc<AtomicU64>,
    backlog_limit: usize,
    max_inflight: usize,
    backlog_reporter: Option<BacklogReporter>,
    sender: Mutex<Option<Sender<BackendCommand>>>,
    thread: Mutex<Option<thread::JoinHandle<()>>>,
}

enum BackendCommand {
    Submit(WriteRequest),
    Close,
}

impl LinuxFileWriter {
    pub(crate) fn new(
        file: File,
        options: ParallelFileWriterOptions,
    ) -> Result<Self, ParallelFileWriterError> {
        let max_inflight = options.worker_threads.max(1);
        let ring_entries = (max_inflight * 2).next_power_of_two().clamp(2, 1024) as u32;
        let (tx, rx) = mpsc::channel();
        let state = Arc::new((
            Mutex::new(SharedState {
                pending_bytes: 0,
                inflight_ops: 0,
                closed: false,
                stopped: false,
            }),
            Condvar::new(),
        ));
        let error = Arc::new(Mutex::new(None));
        let backlog_bytes = Arc::new(AtomicU64::new(0));
        let state_ref = Arc::clone(&state);
        let error_ref = Arc::clone(&error);
        let backlog_ref = Arc::clone(&backlog_bytes);
        let backlog_reporter = options.backlog_reporter.clone();
        let write_reporter = options.write_reporter.clone();
        let handle = thread::spawn(move || {
            backend_loop(
                file,
                ring_entries,
                max_inflight,
                rx,
                state_ref,
                error_ref,
                backlog_ref,
                backlog_reporter,
                write_reporter,
            );
        });

        Ok(Self {
            state,
            error,
            backlog_bytes,
            backlog_limit: options.max_backlog_bytes.max(1),
            max_inflight,
            backlog_reporter: options.backlog_reporter,
            sender: Mutex::new(Some(tx)),
            thread: Mutex::new(Some(handle)),
        })
    }

    fn check_error(&self) -> Result<(), ParallelFileWriterError> {
        let mut slot = self
            .error
            .lock()
            .map_err(|_| io::Error::other("linux writer error slot poisoned"))?;
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

impl FileWriterBackend for LinuxFileWriter {
    fn submit(&self, offset: u64, data: Vec<u8>) -> Result<(), ParallelFileWriterError> {
        self.check_error()?;
        let (lock, cv) = &*self.state;
        let mut state = lock
            .lock()
            .map_err(|_| io::Error::other("linux writer state poisoned"))?;
        while (state.pending_bytes >= self.backlog_limit || state.inflight_ops >= self.max_inflight)
            && !state.stopped
            && !state.closed
        {
            state = cv
                .wait(state)
                .map_err(|_| io::Error::other("linux writer state poisoned"))?;
            self.check_error()?;
        }
        if state.stopped || state.closed {
            self.check_error()?;
            return Err(ParallelFileWriterError::Closed);
        }
        state.pending_bytes = state.pending_bytes.saturating_add(data.len());
        state.inflight_ops = state.inflight_ops.saturating_add(1);
        self.update_backlog(state.pending_bytes);
        drop(state);

        let tx = self
            .sender
            .lock()
            .map_err(|_| io::Error::other("linux writer sender poisoned"))?
            .clone()
            .ok_or(ParallelFileWriterError::Closed)?;
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        tx.send(BackendCommand::Submit(WriteRequest {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            offset,
            data,
        }))
        .map_err(|_| ParallelFileWriterError::Closed)
    }

    fn backlog_bytes(&self) -> u64 {
        self.backlog_bytes.load(Ordering::Relaxed)
    }

    fn drain(&self) -> Result<(), ParallelFileWriterError> {
        {
            let mut sender = self
                .sender
                .lock()
                .map_err(|_| io::Error::other("linux writer sender poisoned"))?;
            if let Some(tx) = sender.take() {
                let _ = tx.send(BackendCommand::Close);
            }
        }
        {
            let (lock, cv) = &*self.state;
            let mut state = lock
                .lock()
                .map_err(|_| io::Error::other("linux writer state poisoned"))?;
            state.closed = true;
            while state.inflight_ops > 0 && !state.stopped {
                state = cv
                    .wait(state)
                    .map_err(|_| io::Error::other("linux writer state poisoned"))?;
            }
        }
        if let Ok(mut handle) = self.thread.lock() {
            if let Some(handle) = handle.take() {
                let _ = handle.join();
            }
        }
        self.check_error()
    }
}

fn backend_loop(
    file: File,
    ring_entries: u32,
    max_inflight: usize,
    rx: Receiver<BackendCommand>,
    state_ref: Arc<(Mutex<SharedState>, Condvar)>,
    error_ref: Arc<Mutex<Option<ParallelFileWriterError>>>,
    backlog_bytes: Arc<AtomicU64>,
    backlog_reporter: Option<BacklogReporter>,
    write_reporter: Option<WriteReporter>,
) {
    let mut ring = match IoUring::new(ring_entries) {
        Ok(ring) => ring,
        Err(err) => {
            if let Ok(mut slot) = error_ref.lock() {
                *slot = Some(err.into());
            }
            return;
        }
    };
    let fd = types::Fd(file.as_raw_fd());
    let mut inflight: HashMap<u64, WriteRequest> = HashMap::new();
    let mut pending: VecDeque<WriteRequest> = VecDeque::new();
    let mut closing = false;

    loop {
        while inflight.len() < max_inflight {
            let next = if let Some(req) = pending.pop_front() {
                Some(req)
            } else if closing {
                None
            } else {
                match rx.try_recv() {
                    Ok(BackendCommand::Submit(req)) => Some(req),
                    Ok(BackendCommand::Close) => {
                        closing = true;
                        None
                    }
                    Err(TryRecvError::Empty) => None,
                    Err(TryRecvError::Disconnected) => {
                        closing = true;
                        None
                    }
                }
            };
            let Some(req) = next else {
                break;
            };
            let entry = opcode::Write::new(
                fd,
                req.data.as_ptr(),
                u32::try_from(req.data.len()).unwrap_or(u32::MAX),
            )
            .offset(req.offset)
            .build()
            .user_data(req.id);
            unsafe {
                if ring.submission().push(&entry).is_err() {
                    pending.push_front(req);
                    break;
                }
            }
            inflight.insert(req.id, req);
        }

        if !inflight.is_empty() {
            if let Err(err) = ring.submit_and_wait(1) {
                store_backend_error(&state_ref, &error_ref, err.into());
                return;
            }
            let cq = ring.completion();
            for cqe in cq {
                let id = cqe.user_data();
                let result = cqe.result();
                let Some(req) = inflight.remove(&id) else {
                    continue;
                };
                if result < 0 {
                    let err = io::Error::from_raw_os_error(-result);
                    store_backend_error(&state_ref, &error_ref, err.into());
                    return;
                }
                let written = usize::try_from(result).unwrap_or(usize::MAX);
                if written != req.data.len() {
                    let mut remaining = req.data;
                    remaining.drain(..written.min(remaining.len()));
                    pending.push_front(WriteRequest {
                        id,
                        offset: req.offset.saturating_add(u64::try_from(written).unwrap_or(u64::MAX)),
                        data: remaining,
                    });
                } else {
                    complete_write(
                        &state_ref,
                        &backlog_bytes,
                        &backlog_reporter,
                        &write_reporter,
                        req.data.len(),
                    );
                }
            }
            continue;
        }

        if closing {
            return;
        }

        match rx.recv_timeout(Duration::from_millis(10)) {
            Ok(BackendCommand::Submit(req)) => pending.push_back(req),
            Ok(BackendCommand::Close) => closing = true,
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => closing = true,
        }
    }
}

fn complete_write(
    state_ref: &Arc<(Mutex<SharedState>, Condvar)>,
    backlog_bytes: &Arc<AtomicU64>,
    backlog_reporter: &Option<BacklogReporter>,
    write_reporter: &Option<WriteReporter>,
    bytes: usize,
) {
    let (lock, cv) = &**state_ref;
    if let Ok(mut state) = lock.lock() {
        state.pending_bytes = state.pending_bytes.saturating_sub(bytes);
        state.inflight_ops = state.inflight_ops.saturating_sub(1);
        let backlog = u64::try_from(state.pending_bytes).unwrap_or(u64::MAX);
        backlog_bytes.store(backlog, Ordering::Relaxed);
        if let Some(reporter) = backlog_reporter {
            reporter(backlog);
        }
        if let Some(reporter) = write_reporter {
            reporter(u64::try_from(bytes).unwrap_or(u64::MAX));
        }
        cv.notify_all();
    }
}

fn store_backend_error(
    state_ref: &Arc<(Mutex<SharedState>, Condvar)>,
    error_ref: &Arc<Mutex<Option<ParallelFileWriterError>>>,
    error: ParallelFileWriterError,
) {
    if let Ok(mut slot) = error_ref.lock() {
        if slot.is_none() {
            *slot = Some(error);
        }
    }
    let (lock, cv) = &**state_ref;
    if let Ok(mut state) = lock.lock() {
        state.stopped = true;
        cv.notify_all();
    }
}
