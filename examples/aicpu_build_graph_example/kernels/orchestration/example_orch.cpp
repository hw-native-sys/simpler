/**
 * AICPU-build-graph orchestration (host side).
 *
 * This function runs on the host during Runtime.initialize() and is responsible
 * for device memory setup only. It marshals a device-pointer payload into
 * `runtime->orch_args[]` for the AICPU-side builder to consume.
 *
 * Graph building happens on device in build_graph_aicpu(Runtime*).
 */

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "runtime.h"

extern "C" {

namespace {
std::vector<uint8_t> read_file_bytes(const char* path) {
    if (path == nullptr || path[0] == '\0') {
        return {};
    }
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        return {};
    }
    ifs.seekg(0, std::ios::end);
    std::streamoff size = ifs.tellg();
    if (size <= 0) {
        return {};
    }
    ifs.seekg(0, std::ios::beg);

    std::vector<uint8_t> buf(static_cast<size_t>(size));
    if (!ifs.read(reinterpret_cast<char*>(buf.data()), size)) {
        return {};
    }
    return buf;
}
}  // namespace

int prepare_example_graph(Runtime* runtime, uint64_t* args, int arg_count) {
    // Expected args: [host_a, host_b, host_f, size_a, size_b, size_f, SIZE]
    if (runtime == nullptr || args == nullptr || arg_count < 7) {
        std::cerr << "prepare_example_graph: invalid args\n";
        return -1;
    }

    void* host_a = reinterpret_cast<void*>(args[0]);
    void* host_b = reinterpret_cast<void*>(args[1]);
    void* host_f = reinterpret_cast<void*>(args[2]);
    size_t size_a = static_cast<size_t>(args[3]);
    size_t size_b = static_cast<size_t>(args[4]);
    size_t size_f = static_cast<size_t>(args[5]);
    int SIZE = static_cast<int>(args[6]);

    if (host_a == nullptr || host_b == nullptr || host_f == nullptr || SIZE <= 0) {
        std::cerr << "prepare_example_graph: null host ptr or invalid SIZE\n";
        return -1;
    }

    // Allocate device tensors and copy inputs.
    void* dev_a = runtime->host_api.device_malloc(size_a);
    void* dev_b = runtime->host_api.device_malloc(size_b);
    void* dev_f = runtime->host_api.device_malloc(size_f);
    if (!dev_a || !dev_b || !dev_f) {
        std::cerr << "prepare_example_graph: device_malloc failed\n";
        return -1;
    }
    runtime->record_device_alloc(dev_a);
    runtime->record_device_alloc(dev_b);
    runtime->record_device_alloc(dev_f);
    runtime->host_api.copy_to_device(dev_a, host_a, size_a);
    runtime->host_api.copy_to_device(dev_b, host_b, size_b);

    // Output tensor copy-back during finalize.
    runtime->record_tensor_pair(host_f, dev_f, size_f);

    // Allocate intermediate tensors (c, d, e).
    size_t bytes = static_cast<size_t>(SIZE) * sizeof(float);
    void* dev_c = runtime->host_api.device_malloc(bytes);
    void* dev_d = runtime->host_api.device_malloc(bytes);
    void* dev_e = runtime->host_api.device_malloc(bytes);
    if (!dev_c || !dev_d || !dev_e) {
        std::cerr << "prepare_example_graph: intermediate malloc failed\n";
        return -1;
    }
    runtime->record_device_alloc(dev_c);
    runtime->record_device_alloc(dev_d);
    runtime->record_device_alloc(dev_e);

    // Marshal device pointers for AICPU builder:
    // orch_args = [dev_a, dev_b, dev_c, dev_d, dev_e, dev_f, SIZE]
    runtime->orch_argc = 7;
    runtime->orch_args[0] = reinterpret_cast<uint64_t>(dev_a);
    runtime->orch_args[1] = reinterpret_cast<uint64_t>(dev_b);
    runtime->orch_args[2] = reinterpret_cast<uint64_t>(dev_c);
    runtime->orch_args[3] = reinterpret_cast<uint64_t>(dev_d);
    runtime->orch_args[4] = reinterpret_cast<uint64_t>(dev_e);
    runtime->orch_args[5] = reinterpret_cast<uint64_t>(dev_f);
    runtime->orch_args[6] = static_cast<uint64_t>(SIZE);

    // Provide AICPU-side orchestration plugin bytes (.so) so the builder can dlopen+dlsym it on device.
    //
    // This decouples the graph-building program from the runtime binary: updating the orchestration
    // only requires shipping a small plugin `.so`, rather than relinking/reuploading the full runtime.
    const char* aicpu_orch_path = std::getenv("PTO_AICPU_ORCH_SO");
    const char* aicpu_orch_func = std::getenv("PTO_AICPU_ORCH_FUNC");  // optional

    std::vector<uint8_t> so_bytes = read_file_bytes(aicpu_orch_path);
    if (so_bytes.empty()) {
        std::cerr << "prepare_example_graph: failed to read PTO_AICPU_ORCH_SO="
                  << (aicpu_orch_path ? aicpu_orch_path : "<unset>") << "\n";
        return -1;
    }

    if (!runtime->try_set_aicpu_orch_so(so_bytes.data(), so_bytes.size())) {
        std::cerr << "prepare_example_graph: failed to embed AICPU orchestration plugin into Runtime "
                     "(size="
                  << so_bytes.size() << " bytes, max=" << RUNTIME_MAX_AICPU_ORCH_SO_SIZE << " bytes)\n";
        return -1;
    }
    memset(runtime->aicpu_orch_func_name, 0, sizeof(runtime->aicpu_orch_func_name));
    if (aicpu_orch_func && aicpu_orch_func[0] != '\0') {
        strncpy(runtime->aicpu_orch_func_name, aicpu_orch_func, sizeof(runtime->aicpu_orch_func_name) - 1);
    } else {
        strncpy(runtime->aicpu_orch_func_name, "build_graph_aicpu", sizeof(runtime->aicpu_orch_func_name) - 1);
    }

    std::cout << "prepare_example_graph: loaded AICPU orch plugin " << (aicpu_orch_path ? aicpu_orch_path : "<unset>")
              << " (file_bytes=" << so_bytes.size() << ", embedded_bytes=" << runtime->get_aicpu_orch_so_size()
              << "), func=" << runtime->aicpu_orch_func_name << "\n";

    return 0;
}

}  // extern "C"
