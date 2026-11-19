#include "sample_support.h"
#include <aclnnop/aclnn_sync_batch_norm_gather_stats.h>
#include <dlfcn.h>

int main(int argc, char** argv)
{
    if (argc != 9) {
        std::cerr << "usage: runner ROOT DEVICE C f16|f32 i32|i64 MOMENTUM EPS WORLD\n";
        return 2;
    }
    try {
        const std::string root = argv[1], dtypeName = argv[4], countName = argv[5];
        const int device = std::stoi(argv[2]);
        const int64_t c = std::stoll(argv[3]), world = std::stoll(argv[8]);
        if (c <= 0 || world <= 0 || (dtypeName != "f16" && dtypeName != "f32") ||
            (countName != "i32" && countName != "i64")) {
            throw std::runtime_error("invalid arguments");
        }
        Dl_info info{};
        if (!dladdr(reinterpret_cast<void*>(aclnnSyncBatchNormGatherStatsGetWorkspaceSize), &info) ||
            !info.dli_fname) {
            throw std::runtime_error("cannot identify ACLNN library");
        }
        std::cout << "ACLNN implementation: " << info.dli_fname << std::endl;
        const aclDataType dtype = dtypeName == "f16" ? ACL_FLOAT16 : ACL_FLOAT;
        const aclDataType countType = countName == "i64" ? ACL_INT64 : ACL_INT32;
        const size_t elemBytes = dtypeName == "f16" ? 2 : 4;
        const size_t countBytes = countName == "i64" ? 8 : 4;
        check(aclInit(nullptr));
        check(aclrtSetDevice(device));
        aclrtStream stream;
        check(aclrtCreateStream(&stream));
        {
            Tensor sum({world, c}, dtype, world * c * elemBytes, "total_sum.bin", true, false, root);
            Tensor sq({world, c}, dtype, world * c * elemBytes, "total_square_sum.bin", true, false, root);
            Tensor count({1}, countType, countBytes, "sample_count.bin", true, false, root);
            Tensor mean({c}, dtype, c * elemBytes, "mean.bin", true, false, root);
            Tensor var({c}, dtype, c * elemBytes, "variance.bin", true, false, root);
            Tensor outMean({c}, dtype, c * elemBytes, "batch_mean.bin", false, false, root);
            Tensor outInv({c}, dtype, c * elemBytes, "batch_invstd.bin", false, false, root);
            uint64_t workspaceSize = 0;
            aclOpExecutor* executor = nullptr;
            check(aclnnSyncBatchNormGatherStatsGetWorkspaceSize(
                sum.tensor, sq.tensor, count.tensor, mean.tensor, var.tensor,
                std::stof(argv[6]), std::stof(argv[7]), outMean.tensor, outInv.tensor,
                &workspaceSize, &executor));
            void* workspace = nullptr;
            if (workspaceSize) {
                check(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
            }
            check(aclnnSyncBatchNormGatherStats(workspace, workspaceSize, executor, stream));
            check(aclrtSynchronizeStream(stream));
            outMean.save(root);
            outInv.save(root);
            mean.file = "new_mean.bin";
            var.file = "new_variance.bin";
            mean.save(root);
            var.save(root);
            if (workspace) {
                check(aclrtFree(workspace));
            }
        }
        check(aclrtDestroyStream(stream));
        check(aclrtResetDevice(device));
        check(aclFinalize());
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
