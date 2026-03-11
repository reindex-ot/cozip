use std::fs::File;
use std::io;
use std::mem::zeroed;
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use std::ptr::{null_mut};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use windows_sys::Win32::Foundation::{
    CloseHandle, GetLastError, HANDLE, INVALID_HANDLE_VALUE, ERROR_IO_PENDING,
};
use windows_sys::Win32::Storage::FileSystem::{
    FILE_FLAG_OVERLAPPED, FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE, ReOpenFile,
    WriteFile,
};
use windows_sys::Win32::System::IO::{
    CreateIoCompletionPort, GetQueuedCompletionStatus, OVERLAPPED, PostQueuedCompletionStatus,
};

use super::FileWriterBackend;
use crate::error::ParallelFileWriterError;
use crate::options::{BacklogReporter, ParallelFileWriterOptions, WriteReporter};

struct SharedState {
    pending_bytes: usize,
    inflight_ops: usize,
    closed: bool,
    stopped: bool,
}

struct CompletionPort(HANDLE);

unsafe impl Send for CompletionPort {}
unsafe impl Sync for CompletionPort {}

impl Drop for CompletionPort {
    fn drop(&mut self) {
        unsafe {
            CloseHandle(self.0);
        }
    }
}

#[repr(C)]
struct WriteOp {
    overlapped: OVERLAPPED,
    data: Vec<u8>,
}

pub(crate) struct WindowsFileWriter {
    _file: File,
    io_file: File,
    port: Arc<CompletionPort>,
    state: Arc<(Mutex<SharedState>, Condvar)>,
    error: Arc<Mutex<Option<ParallelFileWriterError>>>,
    backlog_bytes: Arc<AtomicU64>,
    backlog_limit: usize,
    max_inflight: usize,
    backlog_reporter: Option<BacklogReporter>,
    completion_thread: Mutex<Option<thread::JoinHandle<()>>>,
}

impl WindowsFileWriter {
    pub(crate) fn new(
        file: File,
        options: ParallelFileWriterOptions,
    ) -> Result<Self, ParallelFileWriterError> {
        let io_file = reopen_overlapped(&file)?;
        let port_handle = unsafe {
            CreateIoCompletionPort(
                io_file.as_raw_handle() as HANDLE,
                null_mut(),
                0,
                options.worker_threads.max(1) as u32,
            )
        };
        if port_handle.is_null() {
            return Err(io::Error::last_os_error().into());
        }
        let port = Arc::new(CompletionPort(port_handle));
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
        let backlog_reporter = options.backlog_reporter.clone();
        let write_reporter = options.write_reporter.clone();
        let state_ref = Arc::clone(&state);
        let error_ref = Arc::clone(&error);
        let backlog_ref = Arc::clone(&backlog_bytes);
        let port_ref = Arc::clone(&port);
        let completion_thread = thread::spawn(move || {
            completion_loop(
                port_ref,
                state_ref,
                error_ref,
                backlog_ref,
                backlog_reporter,
                write_reporter,
            );
        });

        Ok(Self {
            _file: file,
            io_file,
            port,
            state,
            error,
            backlog_bytes,
            backlog_limit: options.max_backlog_bytes.max(1),
            max_inflight: options.worker_threads.max(1),
            backlog_reporter: options.backlog_reporter,
            completion_thread: Mutex::new(Some(completion_thread)),
        })
    }

    fn check_error(&self) -> Result<(), ParallelFileWriterError> {
        let mut slot = self
            .error
            .lock()
            .map_err(|_| io::Error::other("windows writer error slot poisoned"))?;
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

impl FileWriterBackend for WindowsFileWriter {
    fn submit(&self, offset: u64, data: Vec<u8>) -> Result<(), ParallelFileWriterError> {
        self.check_error()?;
        let (lock, cv) = &*self.state;
        let mut state = lock
            .lock()
            .map_err(|_| io::Error::other("windows writer state poisoned"))?;
        while (state.pending_bytes >= self.backlog_limit || state.inflight_ops >= self.max_inflight)
            && !state.stopped
            && !state.closed
        {
            state = cv
                .wait(state)
                .map_err(|_| io::Error::other("windows writer state poisoned"))?;
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

        let mut op = Box::new(WriteOp {
            overlapped: unsafe { zeroed() },
            data,
        });
        op.overlapped.Anonymous.Anonymous.Offset = offset as u32;
        op.overlapped.Anonymous.Anonymous.OffsetHigh = (offset >> 32) as u32;
        let len_u32 =
            u32::try_from(op.data.len()).map_err(|_| ParallelFileWriterError::NumericOverflow)?;
        let op_ptr = Box::into_raw(op);
        let write_ok = unsafe {
            WriteFile(
                self.io_file.as_raw_handle() as HANDLE,
                (*op_ptr).data.as_ptr(),
                len_u32,
                null_mut(),
                &mut (*op_ptr).overlapped,
            )
        };
        if write_ok == 0 {
            let err = unsafe { GetLastError() };
            if err != ERROR_IO_PENDING {
                unsafe {
                    drop(Box::from_raw(op_ptr));
                }
                let mut state = lock
                    .lock()
                    .map_err(|_| io::Error::other("windows writer state poisoned"))?;
                state.pending_bytes = state.pending_bytes.saturating_sub(len_u32 as usize);
                state.inflight_ops = state.inflight_ops.saturating_sub(1);
                self.update_backlog(state.pending_bytes);
                cv.notify_all();
                return Err(io::Error::from_raw_os_error(err as i32).into());
            }
        }
        Ok(())
    }

    fn backlog_bytes(&self) -> u64 {
        self.backlog_bytes.load(Ordering::Relaxed)
    }

    fn drain(&self) -> Result<(), ParallelFileWriterError> {
        {
            let (lock, cv) = &*self.state;
            let mut state = lock
                .lock()
                .map_err(|_| io::Error::other("windows writer state poisoned"))?;
            state.closed = true;
            unsafe {
                PostQueuedCompletionStatus(self.port.0, 0, 0, null_mut());
            }
            while state.inflight_ops > 0 && !state.stopped {
                state = cv
                    .wait(state)
                    .map_err(|_| io::Error::other("windows writer state poisoned"))?;
            }
        }
        unsafe {
            PostQueuedCompletionStatus(self.port.0, 0, 0, null_mut());
        }
        if let Ok(mut handle) = self.completion_thread.lock() {
            if let Some(handle) = handle.take() {
                let _ = handle.join();
            }
        }
        self.check_error()
    }
}

fn completion_loop(
    port: Arc<CompletionPort>,
    state_ref: Arc<(Mutex<SharedState>, Condvar)>,
    error_ref: Arc<Mutex<Option<ParallelFileWriterError>>>,
    backlog_bytes: Arc<AtomicU64>,
    backlog_reporter: Option<BacklogReporter>,
    write_reporter: Option<WriteReporter>,
) {
    loop {
        let mut transferred = 0u32;
        let mut completion_key = 0usize;
        let mut overlapped = null_mut();
        let ok = unsafe {
            GetQueuedCompletionStatus(
                port.0,
                &mut transferred,
                &mut completion_key,
                &mut overlapped,
                u32::MAX,
            )
        };
        if overlapped.is_null() {
            let (lock, cv) = &*state_ref;
            let state = match lock.lock() {
                Ok(guard) => guard,
                Err(_) => return,
            };
            if state.closed && state.inflight_ops == 0 {
                cv.notify_all();
                return;
            }
            continue;
        }

        let op = unsafe { Box::from_raw(overlapped as *mut WriteOp) };
        let op_len = op.data.len();
        let mut completion_error = None;
        if ok == 0 {
            completion_error = Some(io::Error::last_os_error().into());
        } else if transferred as usize != op_len {
            completion_error = Some(ParallelFileWriterError::Io(io::Error::new(
                io::ErrorKind::WriteZero,
                "windows overlapped write completed partially",
            )));
        }

        let (lock, cv) = &*state_ref;
        if let Ok(mut state) = lock.lock() {
            state.pending_bytes = state.pending_bytes.saturating_sub(op_len);
            state.inflight_ops = state.inflight_ops.saturating_sub(1);
            let bytes64 = u64::try_from(state.pending_bytes).unwrap_or(u64::MAX);
            backlog_bytes.store(bytes64, Ordering::Relaxed);
            if let Some(reporter) = &backlog_reporter {
                reporter(bytes64);
            }
            if let Some(error) = completion_error {
                state.stopped = true;
                if let Ok(mut slot) = error_ref.lock() {
                    if slot.is_none() {
                        *slot = Some(error);
                    }
                }
            } else if let Some(reporter) = &write_reporter {
                reporter(u64::try_from(op_len).unwrap_or(u64::MAX));
            }
            cv.notify_all();
        }
    }
}

fn reopen_overlapped(file: &File) -> Result<File, ParallelFileWriterError> {
    let handle = unsafe {
        ReOpenFile(
            file.as_raw_handle() as HANDLE,
            0x8000_0000u32 | 0x4000_0000u32,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            FILE_FLAG_OVERLAPPED,
        )
    };
    if handle == INVALID_HANDLE_VALUE {
        return Err(io::Error::last_os_error().into());
    }
    let owned = unsafe { OwnedHandle::from_raw_handle(handle as _) };
    Ok(File::from(owned))
}
