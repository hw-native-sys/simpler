# Installation and Runtime Environment

> **Disclaimer**: due to scarce hardware resources, this repository **only
> tracks and supports the two environments listed below** (A3 and A5).
> Other driver / CANN versions may work, but they are outside our CI and
> day-to-day validation, and issues against them will not be supported.

## Supported Environments

A3 and A5 run **different CANN releases** (A3 on 9.0.0 community,
A5 on 9.1.T500), and the NPU drivers are two separate releases as well.
Pick the matching pair for your hardware below.

| Item | A3 | A5 |
| ---- | -- | -- |
| NPU driver | **26.0.rc1** | **25.6.rc1.b108** |
| CANN major version | **9.0.0** | **9.1.T500** |

### How to Check Versions

Read the driver info from `/usr/local/Ascend/driver/version.info` on both
boxes (the file is written by the `.run` installer, does not depend on
`npu-smi`, and is readable on the A5 EVB even though `npu-smi` is
unavailable there):

```bash
cat /usr/local/Ascend/driver/version.info
# Fields of interest:
#   Version=...                ← driver version
#   compatible_version=...     ← compatible CANN version range
```

## Installation Steps

### 1. Install Driver / Firmware / CANN (system-level, one-time)

**Install order: driver → firmware → CANN.** Refer to the official docs for
the exact commands; this repository deliberately does not duplicate them
(they drift across releases).

- Driver / firmware: pick the right `.run` package for your hardware from
  the [Ascend HDK software download page](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-hdk-pid-252764743/software).
  - **A3 direct download**: [Ascend HDK driver / firmware (A3)](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-hdk-pid-252764743/software/267987649?idAbsPath=fixnode01%7C23710424%7C251366513%7C254884019%7C261408772%7C252764743)
- CANN: install the version that matches your hardware
  (A3 → 9.0.0 community, A5 → 9.1.T500). Both offline and online installs
  are available; this repository primarily uses the offline toolkit +
  standalone driver combination.
  - **A3** — follow the [CANN 9.0.0 install guide — openEuler / netyum](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0030.html?OS=openEuler&InstallType=netyum);
    direct download: [CANN 9.0.0 Community Edition (A3)](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0).
  - **A5** — CANN 9.1.T500.

After installing the driver and firmware you typically need a reboot (or a
kernel-module reload). CANN, when installed to the default location,
creates a `latest` symlink under `/usr/local/Ascend/ascend-toolkit/`.
Once installed, use the commands in "How to Check Versions" above to
confirm `version.info` / `version.cfg` can be read back successfully.

### 2. Install This Repository

Follow the standard flow from
[getting-started.md](getting-started.md#install) — identical for A3 and A5:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --no-build-isolation -e .
```
