#include "sample_support.h"
#include "aclnn_sync_batch_norm_gather_stats.h"

int main(int argc, char** argv)
{
    if (argc != 11) {
        std::cerr << "usage: s007_runner ROOT DEVICE C WORLD COUNT_LEN f16|f32 i32|i64 MOMENTUM EPS rank1|rank2\n";
        return 2;
    }
    try {
        const std::string root = argv[1];
        const int device = std::stoi(argv[2]);
        const int64_t c = std::stoll(argv[3]);
        const int64_t world = std::stoll(argv[4]);
        const int64_t countLen = std::stoll(argv[5]);
        const std::string floatType = argv[6];
        const std::string countType = argv[7];
        const double momentum = std::stod(argv[8]);
        const double eps = std::stod(argv[9]);
        const std::string rank = argv[10];
        if (c <= 0 || world <= 0 || countLen <= 0 ||
            (floatType != "f16" && floatType != "f32") ||
            (countType != "i32" && countType != "i64") ||
            (rank != "rank1" && rank != "rank2") || (rank == "rank1" && world != 1)) {
            throw std::runtime_error("invalid test arguments");
        }
        const aclDataType dtype = floatType == "f16" ? ACL_FLOAT16 : ACL_FLOAT;
        const aclDataType countDtype = countType == "i64" ? ACL_INT64 : ACL_INT32;
        const size_t elemBytes = floatType == "f16" ? 2 : 4;
        const size_t countBytes = countType == "i64" ? 8 : 4;
        check(aclInit(nullptr));
        check(aclrtSetDevice(device));
        aclrtStream stream;
        check(aclrtCreateStream(&stream));
        {
            const std::vector<int64_t> sumShape = rank == "rank1" ? std::vector<int64_t>{c}
                                                                  : std::vector<int64_t>{world, c};
            Tensor sum(sumShape, dtype, world * c * elemBytes, "total_sum.bin", true, false, root);
            Tensor sq(sumShape, dtype, world * c * elemBytes, "total_square_sum.bin", true, false, root);
            Tensor count({countLen}, countDtype, countLen * countBytes, "sample_count.bin", true, false, root);
            Tensor mean({c}, dtype, c * elemBytes, "mean.bin", true, false, root);
            Tensor var({c}, dtype, c * elemBytes, "variance.bin", true, false, root);
            Tensor outMean({c}, dtype, c * elemBytes, "batch_mean.bin", false, false, root);
            Tensor outInv({c}, dtype, c * elemBytes, "batch_invstd.bin", false, false, root);
            Tensor outNewMean({c}, ACL_FLOAT, c * 4, "new_mean.bin", false, false, root);
            Tensor outNewVar({c}, ACL_FLOAT, c * 4, "new_variance.bin", false, false, root);
            uint64_t workspaceSize = 0;
            aclOpExecutor* executor = nullptr;
            check(aclnnSyncBatchNormGatherStatsGetWorkspaceSize(
                sum.tensor, sq.tensor, count.tensor, mean.tensor, var.tensor, momentum, eps,
                outMean.tensor, outInv.tensor, outNewMean.tensor, outNewVar.tensor,
                &workspaceSize, &executor));
            void* workspace = nullptr;
            if (workspaceSize) {
                check(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
            }
            check(aclnnSyncBatchNormGatherStats(workspace, workspaceSize, executor, stream));
            check(aclrtSynchronizeStream(stream));
            outMean.save(root);
            outInv.save(root);
            outNewMean.save(root);
            outNewVar.save(root);
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
