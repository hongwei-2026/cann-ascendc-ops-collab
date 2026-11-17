#include "sample_support.h"
#include "aclnn_sync_batch_norm_gather_stats.h"

int main(int argc, char** argv) {
    try {
        std::string root = argc > 1 ? argv[1] : "../samples/build";
        int device = argc > 2 ? std::stoi(argv[2]) : 0;
        check(aclInit(nullptr));
        check(aclrtSetDevice(device));
        aclrtStream stream;
        check(aclrtCreateStream(&stream));
        {
            Tensor input_0_0({1, 6}, ACL_FLOAT, 24, "total_sum.bin", true, false, root);
            Tensor input_1_0({1, 6}, ACL_FLOAT, 24, "total_square_sum.bin", true, false, root);
            Tensor input_2_0({1}, ACL_INT32, 4, "sample_count.bin", true, false, root);
            Tensor input_3_0({6}, ACL_FLOAT, 24, "mean.bin", true, false, root);
            Tensor input_4_0({6}, ACL_FLOAT, 24, "variance.bin", true, false, root);
            Tensor output_0_0({6}, ACL_FLOAT, 24, "batch_mean.bin", false, false, root);
            Tensor output_1_0({6}, ACL_FLOAT, 24, "batch_invstd.bin", false, false, root);
            Tensor output_2_0({6}, ACL_FLOAT, 24, "new_mean.bin", false, false, root);
            Tensor output_3_0({6}, ACL_FLOAT, 24, "new_variance.bin", false, false, root);
            double attr_0 = static_cast<double>(0.1);
            double attr_1 = static_cast<double>(1e-05);
            uint64_t size = 0;
            aclOpExecutor* executor = nullptr;
            check(aclnnSyncBatchNormGatherStatsGetWorkspaceSize(
                input_0_0.tensor,
                input_1_0.tensor,
                input_2_0.tensor,
                input_3_0.tensor,
                input_4_0.tensor,
                attr_0,
                attr_1,
                output_0_0.tensor,
                output_1_0.tensor,
                output_2_0.tensor,
                output_3_0.tensor,
                &size,
                &executor));
            void* workspace = nullptr;
            if (size)
                check(aclrtMalloc(&workspace, size, ACL_MEM_MALLOC_HUGE_FIRST));
            check(aclnnSyncBatchNormGatherStats(workspace, size, executor, stream));
            check(aclrtSynchronizeStream(stream));
            output_0_0.save(root);
            output_1_0.save(root);
            output_2_0.save(root);
            output_3_0.save(root);
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
