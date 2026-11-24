# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Report the A5 Host-Device link of each selected logical device. Exit 77 for PCIe."""

import ctypes
import sys

ACL_DEV_ATTR_HD_CONNECT_TYPE = 408
_LINK_NAMES = {0: "PCIe", 1: "HCCS", 2: "UB"}


def host_links(devices):
    acl = ctypes.CDLL("libascendcl.so")
    acl.aclInit.argtypes = [ctypes.c_char_p]
    acl.aclrtGetDeviceInfo.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.POINTER(ctypes.c_int64)]
    rc = acl.aclInit(None)
    if rc:
        raise RuntimeError(f"aclInit failed: rc={rc}")
    links = []
    try:
        for device in devices:
            value = ctypes.c_int64(-1)
            rc = acl.aclrtGetDeviceInfo(device, ACL_DEV_ATTR_HD_CONNECT_TYPE, ctypes.byref(value))
            if rc or value.value not in _LINK_NAMES:
                raise RuntimeError(f"ACL_DEV_ATTR_HD_CONNECT_TYPE: device={device}, rc={rc}, value={value.value}")
            link = _LINK_NAMES[value.value]
            print(f"device={device} Host-Device link={link} ({value.value})", flush=True)
            links.append(link)
    finally:
        rc = acl.aclFinalize()
        if rc:
            raise RuntimeError(f"aclFinalize failed: rc={rc}")
    return links


if __name__ == "__main__":
    sys.exit(77 if "PCIe" in host_links(map(int, sys.argv[1:])) else 0)
