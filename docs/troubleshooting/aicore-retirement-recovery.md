# AICore retirement failure and device-owner recovery

An AICore exit ACK is not a completion fence. The retirement owner must
observe this request's nonce, close/read back the window, drain that close,
and publish the post-close release before the Core may return.

## ACK after the retirement deadline

`platform_retire_aicore_group` broadcasts EXIT and sweeps the claimed group
against one shared deadline. A Core not acknowledged by that sweep remains
unclosed and unreleased. The helper returns failure and the scheduler's
exclusive claim remains consumed. A matching ACK written after the helper
returns does not elect another owner or perform close/drain/release.

This is a terminal retirement failure, not an automatic same-generation
retry contract. Normal Core join cannot complete through that path alone.
Reissuing the business kernel, extending its deadline, clearing the claim,
or forcing the release gate is not device recovery. A new request can also
leave the Core waiting on the release of an earlier request.

The rule applies to the shared A5 and A2/A3 retirement helpers. TMR, HBG
legacy and A5 HBG resident callers have distinct ownership and completion
paths; sharing the helper does not prove identical whole-runtime fault
behavior. CPU controls can verify nonce/window/gate state, not CANN's
cancellation or recovery behavior on silicon.

## The Host handoff

The Host waits for its own completion boundaries with the effective
configured timeout. Nonblocking poll is observation only; its caller must
provide bounded monitoring. Neither callback return nor a published error
proves that every Core has returned. A queued successor behind a completion
that never arrives cannot enter merely to inspect the predecessor's error.

If the completion wait fails, the runner stops admission and follows its
existing ownership policy:

- **PROGRAM / owned device:** the runner becomes unusable. Its fatal
  finalize path attempts the owned-device recovery and checks its result
  before retiring the device generation. A failed reset is not permission
  for per-resource RTS cleanup.
- **KERNEL / borrowed device:** the runner becomes unusable and its kernel
  state is poisoned. It performs no device drain or reset. Fatal close
  reports failure, retains the resources for the process lifetime, and
  refuses destruction. The external device owner controls recovery; this
  context has no API that reclaims the quarantined generation after an
  external reset.

A failed thread attachment or an exception escaping the backend drain also
leaves completion unproved. These paths mark the runner unusable without
calling RTS: the calling thread may have a different device/context current.
They preserve the original error and resource references. Retrying ordinary
cleanup is not evidence that the submitted work finished.

Finalizing the native run releases its reservation, not the quarantined
device's code or tables. Callable unregistration refuses the unusable
generation before changing the registry or allocation references. Those
resources remain under the same device-owner recovery policy.

Completed device work and diagnostics failure are separate. A diagnostics
error after proved completion does not by itself quarantine the device.
The backend retains responsibility for graded drain failures whose device
work and queued references have already been settled.

## Evidence boundaries

The deterministic CPU controls exercise the real shared retirement helpers
with simulated registers and publish the current ACK only after the helper
returns through its expired-deadline branch. Native-admission controls use
the real onboard C API and workspace ownership over CPU transport to check
lost attachment, escaping drain exceptions, denied admission, retained
references and diagnostics-only failure.

These controls do not reproduce a hardware hang or the historical CPU
callback watchdog. A CPU watchdog requires its own callback timing and
participant evidence; a Core held at the post-close gate is not that proof.
