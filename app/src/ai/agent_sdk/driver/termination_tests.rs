use std::fs;
use std::time::Duration;

use futures::executor::block_on;
use nix::sys::signal::Signal;
use nix::unistd::Pid;
use signal_hook::low_level::siginfo::{Cause, Sent};
use tempfile::TempDir;

use super::InterruptSignal;

/// Set on the re-executed child so it takes the [`signal_lifecycle_child`] path instead
/// of running the test body again. Its value selects which lifecycle to exercise.
const SIGNAL_CHILD_ENV: &str = "WARP_AGENT_DRIVER_SIGNAL_CHILD";
/// Printed by the child once its interrupt watch is armed, so the parent knows when the
/// first signal can be delivered.
const READY_MARKER: &str = "warp-signal-test: ready";
/// Printed by the child once it starts (and, for `int-hang`, deliberately stalls) its
/// shutdown work, so the parent knows when the second signal is meaningful.
const SHUTDOWN_MARKER: &str = "warp-signal-test: shutdown_started";
/// How long an `int-hang` child stalls before giving up. Only reached when the parent
/// fails to deliver the second signal, in which case exiting keeps the test from
/// leaving an orphaned process behind.
const STUCK_SHUTDOWN_BAILOUT: Duration = Duration::from_secs(60);

fn signal_lifecycle_child() -> ! {
    let kind = std::env::var(SIGNAL_CHILD_ENV).expect("child kind");
    let expected = match kind.as_str() {
        "term" => InterruptSignal::Term,
        "int" | "int-hang" => InterruptSignal::Int,
        other => panic!("unknown child kind {other}"),
    };

    let background = warpui::r#async::executor::Background::default();
    let (signal_rx, _watch) =
        block_on(super::watch_interrupt_signals(&background)).expect("signal watch");
    println!("{READY_MARKER}");

    let observed = block_on(signal_rx).expect("interrupt signal");
    assert_eq!(observed.signal, expected);
    #[cfg(target_os = "linux")]
    {
        let sender = observed.sender.expect("sender process");
        assert_eq!(sender.pid, nix::unistd::getppid().as_raw());
        assert_eq!(sender.uid, nix::unistd::getuid().as_raw());
        assert_eq!(observed.cause, Cause::Sent(Sent::User));
    }

    if kind == "int-hang" {
        println!("{SHUTDOWN_MARKER}");
        std::thread::sleep(STUCK_SHUTDOWN_BAILOUT);
        std::process::exit(1);
    }
    super::emulate_default_and_exit(observed.signal);
}

/// Re-executes this test binary as a child running only `test_name`, delivers
/// `first_sig` (and `second_sig`, once the child reports it is stuck in shutdown), and
/// asserts the child died from `expected_sig`.
///
/// Signal dispositions are process-wide, so the behavior under test is only observable
/// in a dedicated process; `test_name` must therefore be the fully-qualified name of the
/// calling test.
fn spawn_signal_lifecycle_child(
    kind: &str,
    test_name: &str,
    first_sig: Signal,
    second_sig: Option<Signal>,
    expected_sig: Signal,
    expect_stuck_shutdown: bool,
) {
    use std::os::unix::process::ExitStatusExt as _;

    if std::env::var_os(SIGNAL_CHILD_ENV).is_some() {
        signal_lifecycle_child();
    }

    let dir = TempDir::new().unwrap();
    let stdout_path = dir.path().join("stdout");
    let stderr_path = dir.path().join("stderr");
    let mut cmd = command::blocking::Command::new(std::env::current_exe().unwrap());
    cmd.arg(test_name)
        .arg("--exact")
        .arg("--nocapture")
        .env(SIGNAL_CHILD_ENV, kind)
        .env("RUST_TEST_THREADS", "1")
        .stdout(fs::File::create(&stdout_path).unwrap())
        .stderr(fs::File::create(&stderr_path).unwrap());
    for (key, _) in std::env::vars() {
        if key.starts_with("NEXTEST") {
            cmd.env_remove(&key);
        }
    }
    let mut child = cmd.spawn().unwrap();

    let wait_for_marker = |child: &mut std::process::Child, marker: &str| {
        for _ in 0..1_000 {
            if fs::read_to_string(&stdout_path)
                .unwrap_or_default()
                .contains(marker)
            {
                return;
            }
            if let Some(status) = child.try_wait().unwrap() {
                panic!(
                    "signal child exited before printing {marker:?}: status={status:?} \
                     stdout={} stderr={}",
                    fs::read_to_string(&stdout_path).unwrap_or_default(),
                    fs::read_to_string(&stderr_path).unwrap_or_default()
                );
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        panic!(
            "signal child never printed {marker:?}; stdout={} stderr={}",
            fs::read_to_string(&stdout_path).unwrap_or_default(),
            fs::read_to_string(&stderr_path).unwrap_or_default()
        );
    };

    wait_for_marker(&mut child, READY_MARKER);

    let child_pid = Pid::from_raw(i32::try_from(child.id()).unwrap());
    nix::sys::signal::kill(child_pid, first_sig).unwrap();

    if let Some(second_sig) = second_sig {
        wait_for_marker(&mut child, SHUTDOWN_MARKER);
        nix::sys::signal::kill(child_pid, second_sig).unwrap();
    }

    let status = child.wait().unwrap();
    let stdout = fs::read_to_string(&stdout_path).unwrap_or_default();
    assert_eq!(
        status.signal(),
        Some(expected_sig as i32),
        "status={status:?} stdout={stdout} stderr={}",
        fs::read_to_string(&stderr_path).unwrap_or_default()
    );
    assert_eq!(
        stdout.contains(SHUTDOWN_MARKER),
        expect_stuck_shutdown,
        "stdout={stdout}"
    );
}

#[test]
fn sigterm_subprocess_exits_signaled() {
    spawn_signal_lifecycle_child(
        "term",
        "ai::agent_sdk::driver::termination::tests::sigterm_subprocess_exits_signaled",
        Signal::SIGTERM,
        None,
        Signal::SIGTERM,
        false,
    );
}

#[test]
fn sigint_subprocess_exits_signaled() {
    spawn_signal_lifecycle_child(
        "int",
        "ai::agent_sdk::driver::termination::tests::sigint_subprocess_exits_signaled",
        Signal::SIGINT,
        None,
        Signal::SIGINT,
        false,
    );
}

#[test]
fn second_sigint_kills_during_stuck_shutdown() {
    spawn_signal_lifecycle_child(
        "int-hang",
        "ai::agent_sdk::driver::termination::tests::second_sigint_kills_during_stuck_shutdown",
        Signal::SIGINT,
        Some(Signal::SIGINT),
        Signal::SIGINT,
        true,
    );
}

#[test]
fn formats_signal_cause_for_trace_fields() {
    assert_eq!(super::cause_name(Cause::Sent(Sent::User)), "sent_user");
    assert_eq!(super::cause_name(Cause::Unknown), "unknown");
}

#[test]
fn formats_process_arguments_as_a_shell_command_line() {
    let arguments = ["agent", "--prompt", "hello world"].map(Into::into);

    assert_eq!(
        super::format_command_line(&arguments),
        Some("agent --prompt 'hello world'".to_owned())
    );
}

#[test]
fn omits_command_line_when_process_arguments_are_empty() {
    assert_eq!(super::format_command_line(&[]), None);
}

#[test]
fn omits_command_line_for_invalid_sender_pid() {
    assert_eq!(super::resolve_command_line(0), None);
}
