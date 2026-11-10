#include "sample_support.h"
#include "aclnn_confusion_transpose_d.h"

#include <sstream>

namespace {

aclDataType ParseDtype(const std::string& name) {
    if (name == "float32" || name == "float") return ACL_FLOAT;
    if (name == "float16" || name == "half") return ACL_FLOAT16;
    if (name == "bfloat16" || name == "bf16") return ACL_BF16;
    if (name == "int8") return ACL_INT8;
    if (name == "int16") return ACL_INT16;
    if (name == "int32") return ACL_INT32;
    if (name == "int64") return ACL_INT64;
    if (name == "uint8") return ACL_UINT8;
    if (name == "uint16") return ACL_UINT16;
    if (name == "uint32") return ACL_UINT32;
    if (name == "uint64") return ACL_UINT64;
    throw std::runtime_error("unsupported dtype: " + name);
}

size_t DtypeBytes(aclDataType type) {
    switch (type) {
        case ACL_INT8:
        case ACL_UINT8:
            return 1;
        case ACL_FLOAT16:
        case ACL_BF16:
        case ACL_INT16:
        case ACL_UINT16:
            return 2;
        case ACL_FLOAT:
        case ACL_INT32:
        case ACL_UINT32:
            return 4;
        case ACL_INT64:
        case ACL_UINT64:
            return 8;
        default:
            throw std::runtime_error("unknown dtype bytes");
    }
}

std::vector<int64_t> ParseIntList(const std::string& text) {
    std::vector<int64_t> out;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        out.push_back(std::stoll(item));
    }
    if (out.empty()) throw std::runtime_error("empty int list");
    return out;
}

struct CaseMeta {
    std::string dtype = "float32";
    bool transposeFirst = true;
    std::vector<int64_t> inShape{2, 3, 4, 5};
    std::vector<int64_t> perm{0, 2, 1, 3};
    std::vector<int64_t> outShape{2, 4, 3, 5};
};

std::string Trim(std::string s) {
    while (!s.empty() && (s.back() == '\r' || s.back() == '\n' || s.back() == ' ')) s.pop_back();
    size_t i = 0;
    while (i < s.size() && s[i] == ' ') ++i;
    return s.substr(i);
}

CaseMeta LoadMeta(const std::string& root) {
    CaseMeta meta;
    const std::string path = root + "/meta.txt";
    std::ifstream in(path);
    if (!in) return meta; // default sample case
    std::string line;
    while (std::getline(in, line)) {
        line = Trim(line);
        if (line.empty() || line[0] == '#') continue;
        const auto pos = line.find('=');
        if (pos == std::string::npos) continue;
        const std::string key = Trim(line.substr(0, pos));
        const std::string val = Trim(line.substr(pos + 1));
        if (key == "dtype") meta.dtype = val;
        else if (key == "transpose_first") meta.transposeFirst = (val == "1" || val == "true" || val == "True");
        else if (key == "in_shape") meta.inShape = ParseIntList(val);
        else if (key == "perm") meta.perm = ParseIntList(val);
        else if (key == "out_shape") meta.outShape = ParseIntList(val);
    }
    if (meta.perm.size() != meta.outShape.size()) {
        throw std::runtime_error("perm/out_shape rank mismatch");
    }
    return meta;
}

int64_t Numel(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string root = argc > 1 ? argv[1] : "../samples/build";
        int device = argc > 2 ? std::stoi(argv[2]) : 0;
        CaseMeta meta = LoadMeta(root);
        aclDataType xType = ParseDtype(meta.dtype);
        const size_t elemBytes = DtypeBytes(xType);
        const int64_t total = Numel(meta.inShape);
        if (total <= 0 || Numel(meta.outShape) != total) {
            throw std::runtime_error("in/attr-shape numel mismatch");
        }
        // attr `shape`：TF=true 为最终输出；TF=false 为 reshape 中间态，最终为 permute(shape)
        std::vector<int64_t> finalShape = meta.outShape;
        if (!meta.transposeFirst) {
            finalShape.resize(meta.perm.size());
            for (size_t i = 0; i < meta.perm.size(); ++i) {
                finalShape[i] = meta.outShape[static_cast<size_t>(meta.perm[i])];
            }
        }
        const size_t bytes = static_cast<size_t>(total) * elemBytes;

        check(aclInit(nullptr));
        check(aclrtSetDevice(device));
        aclrtStream stream;
        check(aclrtCreateStream(&stream));
        {
            Tensor input_0_0(meta.inShape, xType, bytes, "x.bin", true, false, root);
            Tensor output_0_0(finalShape, xType, bytes, "y.bin", false, false, root);
            auto* attr_0 = aclCreateIntArray(meta.perm.data(), meta.perm.size());
            if (!attr_0) throw std::runtime_error("attribute creation failed");
            auto* attr_1 = aclCreateIntArray(meta.outShape.data(), meta.outShape.size());
            if (!attr_1) throw std::runtime_error("attribute creation failed");
            bool attr_2 = meta.transposeFirst;
            uint64_t size = 0;
            aclOpExecutor* executor = nullptr;
            check(aclnnConfusionTransposeDGetWorkspaceSize(
                input_0_0.tensor,
                attr_0,
                attr_1,
                attr_2,
                output_0_0.tensor,
                &size,
                &executor));
            std::cout << "workspace_bytes=" << size << " numel=" << total << " dtype=" << meta.dtype << std::endl;
            void* workspace = nullptr;
            if (size)
                check(aclrtMalloc(&workspace, size, ACL_MEM_MALLOC_HUGE_FIRST));
            // Profiling uses fresh executors per call; defaults preserve the sample run.
            const char* warmupText = std::getenv("CTD_PERF_WARMUP");
            const char* repeatText = std::getenv("CTD_PERF_REPEAT");
            const int warmup = warmupText ? std::stoi(warmupText) : 0;
            const int repeat = repeatText ? std::stoi(repeatText) : 1;
            if (warmup < 0 || repeat < 1 || warmup > 1000 || repeat > 1000) {
                throw std::runtime_error("invalid CTD_PERF_WARMUP/REPEAT");
            }
            for (int i = 0; i < warmup + repeat; ++i) {
                if (i != 0) {
                    uint64_t nextSize = 0;
                    check(aclnnConfusionTransposeDGetWorkspaceSize(
                        input_0_0.tensor, attr_0, attr_1, attr_2, output_0_0.tensor, &nextSize, &executor));
                    if (nextSize > size) throw std::runtime_error("workspace grew during repeated execution");
                }
                check(aclnnConfusionTransposeD(workspace, size, executor, stream));
                check(aclrtSynchronizeStream(stream));
            }
            std::cout << "perf_warmup=" << warmup << " perf_repeat=" << repeat << std::endl;
            output_0_0.save(root);
            aclDestroyIntArray(attr_0);
            aclDestroyIntArray(attr_1);
            if (workspace)
                check(aclrtFree(workspace));
        }
        check(aclrtDestroyStream(stream));
        check(aclrtResetDevice(device));
        check(aclFinalize());
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 2;
    }
}
