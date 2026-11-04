#include "sample_support.h"
#include "aclnn_ctc_loss_v2.h"

int main(int argc, char** argv) {
    try {
        std::string root = argc > 1 ? argv[1] : "../samples/build";
        int device = argc > 2 ? std::stoi(argv[2]) : 0;
        check(aclInit(nullptr));
        check(aclrtSetDevice(device));
        aclrtStream stream;
        check(aclrtCreateStream(&stream));
        {
            Tensor input_0_0({10, 1, 5}, ACL_FLOAT, 200, "log_probs.bin", true, false, root);
            Tensor input_1_0({1, 8}, ACL_INT32, 32, "targets.bin", true, false, root);
            Tensor input_2_0({1}, ACL_INT32, 4, "input_lengths.bin", true, false, root);
            Tensor input_3_0({1}, ACL_INT32, 4, "target_lengths.bin", true, false, root);
            Tensor output_0_0({1}, ACL_FLOAT, 4, "neg_log_likelihood.bin", false, false, root);
            Tensor output_1_0({1, 10, 17}, ACL_FLOAT, 680, "log_alpha.bin", false, false, root);
            int64_t attr_0 = 0LL;
            char attr_1[] = "mean";
            bool attr_2 = false;
            uint64_t size = 0;
            aclOpExecutor* executor = nullptr;
            check(aclnnCtcLossV2GetWorkspaceSize(
                input_0_0.tensor,
                input_1_0.tensor,
                input_2_0.tensor,
                input_3_0.tensor,
                attr_0,
                attr_1,
                attr_2,
                output_0_0.tensor,
                output_1_0.tensor,
                &size,
                &executor));
            void* workspace = nullptr;
            if (size)
                check(aclrtMalloc(&workspace, size, ACL_MEM_MALLOC_HUGE_FIRST));
            check(aclnnCtcLossV2(workspace, size, executor, stream));
            check(aclrtSynchronizeStream(stream));
            output_0_0.save(root);
            output_1_0.save(root);
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
