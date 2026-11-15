// Q005 diagnostic runner. Synthetic inputs are described by profile_q005.py.
#include "sample_support.h"
#include "aclnn_quant_update_scatter.h"
#include <limits>
#include <memory>

namespace {
int64_t Number(const char* text, int64_t low, int64_t high) {
    size_t end = 0;
    const std::string value(text);
    const int64_t result = std::stoll(value, &end);
    if (end != value.size() || result < low || result > high)
        throw std::runtime_error("invalid numeric argument: " + value);
    return result;
}
size_t Bytes(std::initializer_list<int64_t> dimensions, size_t elementSize) {
    size_t bytes = elementSize;
    for (int64_t dimension : dimensions) {
        if (static_cast<uint64_t>(dimension) > std::numeric_limits<size_t>::max() / bytes)
            throw std::runtime_error("tensor size overflow");
        bytes *= static_cast<size_t>(dimension);
    }
    return bytes;
}
void SaveOutput(Tensor& output, const std::string& root) {
    QUS_ACL_CHECK(aclrtMemcpy(output.host.data(), output.bytes, output.device,
                            output.bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    std::ofstream file(root + "/output/var_out.bin", std::ios::binary);
    if (!file || !file.write(output.host.data(), static_cast<std::streamsize>(output.bytes)))
        throw std::runtime_error("cannot write diagnostic output");
}
}

int main(int argc, char** argv) {
    try {
        if (argc != 15)
            throw std::runtime_error("usage: profile_case ROOT DEVICE B M VAR_ROWS UPDATE_ROWS WIDTH "
                                     "MODE INDEX_BITS ZP RECIPROCAL INPLACE WARMUP REPEAT");
        const std::string root(argv[1]);
        const int device = static_cast<int>(Number(argv[2], 0, 255));
        const int64_t batch = Number(argv[3], 1, INT32_MAX);
        const int64_t middle = Number(argv[4], 1, INT32_MAX);
        const int64_t varRows = Number(argv[5], 1, INT32_MAX);
        const int64_t updateRows = Number(argv[6], 1, varRows);
        const int64_t width = Number(argv[7], 1, INT32_MAX);
        const int mode = static_cast<int>(Number(argv[8], 0, 1));
        const int bits = static_cast<int>(Number(argv[9], 32, 64));
        if (bits != 32 && bits != 64) throw std::runtime_error("INDEX_BITS must be 32 or 64");
        const bool hasZp = Number(argv[10], 0, 1) != 0;
        const bool reciprocal = Number(argv[11], 0, 1) != 0;
        const bool inplace = Number(argv[12], 0, 1) != 0;
        const int64_t warmup = Number(argv[13], 0, 10000);
        const int64_t repeat = Number(argv[14], 1, 5000);
        if (warmup + repeat > 5000) throw std::runtime_error("WARMUP + REPEAT exceeds 5000");
        QUS_ACL_CHECK(aclInit(nullptr));
        QUS_ACL_CHECK(aclrtSetDevice(device));
        aclrtStream stream = nullptr;
        QUS_ACL_CHECK(aclrtCreateStream(&stream));
        {
            Tensor var({batch, middle, varRows, width}, ACL_INT8,
                       Bytes({batch, middle, varRows, width}, 1), "var.bin", true, false, root);
            Tensor indices({batch}, bits == 32 ? ACL_INT32 : ACL_INT64,
                           Bytes({batch}, bits / 8), "indices.bin", true, false, root);
            Tensor updates({batch, middle, updateRows, width}, mode == 0 ? ACL_FLOAT16 : ACL_BF16,
                           Bytes({batch, middle, updateRows, width}, 2), "updates.bin", true, false, root);
            Tensor scales({1, 1, 1, width}, mode == 0 ? ACL_FLOAT : ACL_BF16,
                          Bytes({width}, mode == 0 ? 4 : 2), "quant_scales.bin", true, false, root);
            std::unique_ptr<Tensor> zp;
            if (hasZp)
                zp.reset(new Tensor({1, 1, 1, width}, mode == 0 ? ACL_INT32 : ACL_BF16,
                                    Bytes({width}, mode == 0 ? 4 : 2), "quant_zero_points.bin", true, false, root));
            std::unique_ptr<Tensor> separateOutput;
            if (!inplace)
                separateOutput.reset(new Tensor({batch, middle, varRows, width}, ACL_INT8,
                                                 var.bytes, "var_out.bin", false, false, root));
            Tensor& output = inplace ? var : *separateOutput;
            void* workspace = nullptr;
            uint64_t capacity = 0;
            for (int64_t iteration = 0; iteration < warmup + repeat; ++iteration) {
                // GetWorkspaceSize returns a fresh, single-use executor on each call.
                uint64_t required = 0;
                aclOpExecutor* executor = nullptr;
                char reduce[] = "update";
                QUS_ACL_CHECK(aclnnQuantUpdateScatterGetWorkspaceSize(
                    var.tensor, indices.tensor, updates.tensor, scales.tensor, zp ? zp->tensor : nullptr,
                    reduce, -2, -1, reciprocal, output.tensor, &required, &executor));
                if (required > capacity) {
                    if (workspace) QUS_ACL_CHECK(aclrtFree(workspace));
                    workspace = nullptr;
                    QUS_ACL_CHECK(aclrtMalloc(&workspace, required, ACL_MEM_MALLOC_HUGE_FIRST));
                    capacity = required;
                }
                QUS_ACL_CHECK(aclnnQuantUpdateScatter(workspace, required, executor, stream));
                QUS_ACL_CHECK(aclrtSynchronizeStream(stream));
            }
            SaveOutput(output, root);
            if (workspace) QUS_ACL_CHECK(aclrtFree(workspace));
            std::cout << "Q005 completed: warmup=" << warmup << " repeat=" << repeat
                      << " inplace=" << inplace << " mode=" << mode << " index_bits=" << bits << '\n';
        }
        QUS_ACL_CHECK(aclrtDestroyStream(stream));
        QUS_ACL_CHECK(aclrtResetDevice(device));
        QUS_ACL_CHECK(aclFinalize());
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
