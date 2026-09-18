//! Termination plumbing for [`AgentDriver`](super::AgentDriver): naming the reason a run
//! stopped, and watching for the Unix interrupts that abort an in-progress run so a
//! handoff snapshot can be saved before the process dies.

#[cfg(unix)]
use std::io;
#[cfg(unix)]
use std::sync::Arc;
#[cfg(unix)]
use std::sync::atomic::AtomicBool;

#[cfg(unix)]
use futures::StreamExt as _;
#[cfg(unix)]
use futures::channel::oneshot;
#[cfg(unix)]
use signal_hook::consts::{SIGINT, SIGTERM};
#[cfg(unix)]
use signal_hook::iterator::exfiltrator::WithOrigin;
#[cfg(unix)]
use signal_hook::low_level::siginfo::{Cause, Chld, Process, Sent};
#[cfg(unix)]
use signal_hook::{SigId, flag};
#[cfg(unix)]
use signal_hook_tokio::SignalsInfo;
#[cfg(unix)]
use warpui::r#async::executor::{Background, BackgroundTask};

/// Unix signal that aborts an in-progress agent run so a handoff snapshot can be
/// saved before the default terminate disposition is restored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum InterruptSignal {
    /// SIGTERM from instance teardown, container stop, or worker termination.
    #[cfg_attr(not(unix), allow(dead_code))]
    Term,
    /// SIGINT from Ctrl-C.
    #[cfg_attr(not(unix), allow(dead_code))]
    Int,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ObservedSignal {
    pub(super) signal: InterruptSignal,
    /// Kernel-reported process metadata. For SIGINT and SIGTERM, this identifies the sender when
    /// the platform supplies it; kernel-originated signals and some platforms omit it.
    #[cfg(unix)]
    sender: Option<Process>,
    #[cfg(unix)]
    cause: Cause,
}
#[cfg(unix)]
impl InterruptSignal {
    fn as_raw(self) -> libc::c_int {
        match self {
            Self::Term => SIGTERM,
            Self::Int => SIGINT,
        }
    }

    fn from_raw(raw: libc::c_int) -> Option<Self> {
        match raw {
            SIGTERM => Some(Self::Term),
            SIGINT => Some(Self::Int),
            _ => None,
        }
    }
}

#[cfg(unix)]
impl ObservedSignal {
    fn from_origin(origin: signal_hook::low_level::siginfo::Origin) -> Option<Self> {
        Some(Self {
            signal: InterruptSignal::from_raw(origin.signal)?,
            sender: origin.process,
            cause: origin.cause,
        })
    }
}
/// Why `run_internal` stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RunEndCause {
    /// `run_internal` returned on its own (success or a non-signal error).
    Completed,
    /// `WARP_SANDBOX_DEADLINE` warning window fired.
    SandboxDeadline,
    /// A Unix interrupt arrived while the run was still in progress.
    Signal(ObservedSignal),
}

/// The interrupt handlers registered by [`watch_interrupt_signals`], plus the background
/// task waiting on their delivery.
///
/// Torn down through [`unregister`](Self::unregister) when the run ends on its own. On
/// the interrupt path the handlers deliberately stay registered instead, so a second
/// signal can still terminate the process if the handoff snapshot gets stuck.
#[cfg(unix)]
pub(super) struct InterruptWatch {
    sig_ids: Vec<SigId>,
    handle: signal_hook_tokio::Handle,
    task: BackgroundTask,
}

#[cfg(unix)]
impl InterruptWatch {
    pub(super) fn unregister(self) {
        self.handle.close();
        self.task.abort();
        for id in self.sig_ids {
            signal_hook::low_level::unregister(id);
        }
    }
}

/// Unregisters the handlers it holds when dropped, so a registration sequence that fails
/// partway through cannot leave handlers installed with nothing watching them.
#[cfg(unix)]
#[derive(Default)]
struct RegisteredHandlers(Vec<SigId>);

#[cfg(unix)]
impl RegisteredHandlers {
    fn push(&mut self, id: SigId) {
        self.0.push(id);
    }

    /// Hands the registrations to the caller, cancelling the unregister-on-drop.
    fn release(mut self) -> Vec<SigId> {
        std::mem::take(&mut self.0)
    }
}

#[cfg(unix)]
impl Drop for RegisteredHandlers {
    fn drop(&mut self) {
        for id in self.0.drain(..) {
            signal_hook::low_level::unregister(id);
        }
    }
}

/// Registers SIGTERM/SIGINT handlers and starts watching for their delivery.
///
/// The returned receiver resolves with the first interrupt observed, or is cancelled if
/// the returned [`InterruptWatch`] is torn down first. Only the first delivery of either
/// signal is reported, so the run gets one chance to save a handoff snapshot; a second
/// delivery bypasses the watch entirely and emulates the default terminate action, so a
/// stuck snapshot cannot trap Ctrl-C or an instance teardown.
#[cfg(unix)]
pub(super) async fn watch_interrupt_signals(
    background: &Background,
) -> io::Result<(oneshot::Receiver<ObservedSignal>, InterruptWatch)> {
    let shutdown_armed = Arc::new(AtomicBool::new(false));
    let mut handlers = RegisteredHandlers::default();
    for signal in [SIGTERM, SIGINT] {
        // Registered before the flag-setting handler below so that the second delivery,
        // and only the second, sees `shutdown_armed` already set and terminates.
        handlers.push(flag::register_conditional_default(
            signal,
            Arc::clone(&shutdown_armed),
        )?);
        handlers.push(flag::register(signal, Arc::clone(&shutdown_armed))?);
    }

    // `Signals` registers with the Tokio reactor, so it has to be built and polled on the
    // background runtime rather than on the driver's own executor.
    let (ready_tx, ready_rx) = oneshot::channel();
    let (signal_tx, signal_rx) = oneshot::channel();
    let task = background.spawn(async move {
        let mut signals = match SignalsInfo::<WithOrigin>::new([SIGTERM, SIGINT]) {
            Ok(signals) => signals,
            Err(error) => {
                let _ = ready_tx.send(Err(error));
                return;
            }
        };
        if ready_tx.send(Ok(signals.handle())).is_err() {
            return;
        }
        // Only the signals registered above are delivered, so the first item maps to one
        // of them. The stream itself only ends once the watch is torn down, at which
        // point nothing is waiting on `signal_tx` any more.
        if let Some(signal) = signals.next().await.and_then(ObservedSignal::from_origin) {
            log::warn!("Received Unix signal {signal:?}");
            let _ = signal_tx.send(signal);
            emit_signal_trace(signal);
        }
    });
    let handle = ready_rx
        .await
        .map_err(|_| io::Error::other("Unix signal watcher task did not start"))??;

    Ok((
        signal_rx,
        InterruptWatch {
            sig_ids: handlers.release(),
            handle,
            task,
        },
    ))
}

#[cfg(unix)]
fn emit_signal_trace(signal: ObservedSignal) {
    let username = signal
        .sender
        .and_then(|process| resolve_username(process.uid));
    let command_line = signal
        .sender
        .and_then(|process| resolve_command_line(process.pid));
    tracing::warn!(
        tags.cloud_agent = true,
        signal = signal_name(signal.signal),
        signal.cause = cause_name(signal.cause),
        signal.sender.pid = signal.sender.map(|process| process.pid),
        signal.sender.uid = signal.sender.map(|process| process.uid),
        signal.sender.username = username,
        signal.sender.command_line = command_line,
        "received unix signal"
    );
}

#[cfg(unix)]
fn signal_name(signal: InterruptSignal) -> &'static str {
    match signal {
        InterruptSignal::Term => "SIGTERM",
        InterruptSignal::Int => "SIGINT",
    }
}

#[cfg(unix)]
fn cause_name(cause: Cause) -> &'static str {
    match cause {
        Cause::Unknown => "unknown",
        Cause::Kernel => "kernel",
        Cause::Sent(sent) => match sent {
            Sent::User => "sent_user",
            Sent::TKill => "sent_tkill",
            Sent::Queue => "sent_queue",
            Sent::MesgQ => "sent_message_queue",
            _ => "sent_other",
        },
        Cause::Chld(child) => match child {
            Chld::Exited => "child_exited",
            Chld::Killed => "child_killed",
            Chld::Dumped => "child_dumped",
            Chld::Trapped => "child_trapped",
            Chld::Stopped => "child_stopped",
            Chld::Continued => "child_continued",
            _ => "child_other",
        },
        _ => "other",
    }
}

#[cfg(unix)]
fn resolve_username(uid: libc::uid_t) -> Option<String> {
    nix::unistd::User::from_uid(nix::unistd::Uid::from_raw(uid))
        .ok()
        .flatten()
        .map(|user| user.name)
}

#[cfg(unix)]
fn resolve_command_line(pid: libc::pid_t) -> Option<String> {
    use sysinfo::{Pid, ProcessRefreshKind, ProcessesToUpdate, System, UpdateKind};

    let pid = u32::try_from(pid).ok().filter(|pid| *pid > 0)?;
    let pid = Pid::from_u32(pid);
    let mut system = System::new();
    system.refresh_processes_specifics(
        ProcessesToUpdate::Some(&[pid]),
        true,
        ProcessRefreshKind::nothing()
            .without_tasks()
            .with_cmd(UpdateKind::Always),
    );
    format_command_line(system.process(pid)?.cmd())
}

#[cfg(unix)]
fn format_command_line(arguments: &[std::ffi::OsString]) -> Option<String> {
    (!arguments.is_empty())
        .then(|| shell_words::join(arguments.iter().map(|argument| argument.to_string_lossy())))
}
#[cfg(unix)]
pub(super) fn emulate_default_and_exit(signal: InterruptSignal) -> ! {
    let _ = signal_hook::low_level::emulate_default_handler(signal.as_raw());
    signal_hook::low_level::abort();
}

#[cfg(all(test, unix))]
#[path = "termination_tests.rs"]
mod tests;
