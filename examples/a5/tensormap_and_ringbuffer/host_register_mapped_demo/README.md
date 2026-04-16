# A5 Host Register Mapped Demo

这个 demo 用来验证 `a5` 平台上的两件事：
- `mallocHostDeviceShareMem(...)` 可以在 Host 侧申请并注册一段 Device 可访问地址
- AIV kernel 可以直接读取并写回这段映射内存

## 本次 a5 修改点

- 在 `src/a5/platform/onboard/host/pto_runtime_c_api.cpp` 中实现了：
  - `mallocHostDeviceShareMem(...)`
  - `freeHostDeviceShareMem(...)`
- 这两个接口的执行顺序和 `a2a3` 保持一致：
  - `GetDevice / SetDevice`
  - `MallocHost / FreeHost`
  - `HostRegister / HostUnregister`
- Python 侧继续复用通用封装：
  - `malloc_host_device_share_mem(...)`
  - `free_host_device_share_mem(...)`
- 新增了 `examples/a5/tensormap_and_ringbuffer/host_register_mapped_demo` 用于硬件验证

## Demo 行为

- Host 通过 `malloc_host_device_share_mem(...)` 拿到：
  - `host_ptr`
  - `mapped_dev_ptr`
- Host 把 `host_ptr` 初始化为 `0, 1, 2, ...`
- orchestration 把 `mapped_dev_ptr` 包成外部 tensor
- kernel 执行：
  - `mapped_host_buffer[i] = mapped_host_buffer[i] + 1`
  - `mapped_out[i] = mapped_host_buffer[i] + 1`
- 运行结束后打印：
  - 初始 Host 数据
  - 执行后 Host 内存数据
  - 普通 output copy-back 数据

## 启动命令

在仓库根目录执行：

```bash
python examples/scripts/run_example.py --build \
  -k examples/a5/tensormap_and_ringbuffer/host_register_mapped_demo/kernels \
  -g examples/a5/tensormap_and_ringbuffer/host_register_mapped_demo/golden.py \
  -p a5 -d 0
```

如果环境已经提前编好了 runtime，也可以去掉 `--build`。

## 结果判断

成功时建议重点看三组日志：
- `a5_host_register_mapped_demo: host_init_data`
- `a5_host_register_mapped_demo: host_data_after_run`
- `a5_host_register_mapped_demo: device_copy_back_data`

理想结果是：
- `host_init_data` 为 `0, 1, 2, ...`
- `host_data_after_run` 为 `1, 2, 3, ...`
- `device_copy_back_data` 也为 `1, 2, 3, ...`
