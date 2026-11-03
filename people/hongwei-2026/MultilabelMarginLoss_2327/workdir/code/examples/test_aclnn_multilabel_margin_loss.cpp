#include "sample_support.h"
#include "aclnn_multilabel_margin_loss.h"

int main(int argc, char** argv) {
    try {
        std::string root = argc > 1 ? argv[1] : "../samples/build";
        int device = argc > 2 ? std::stoi(argv[2]) : 0;
        check(aclInit(nullptr));
        check(aclrtSetDevice(device));
        aclrtStream stream;
        check(aclrtCreateStream(&stream));
        {
            Tensor input_0_0({4, 6}, ACL_FLOAT, 96, "x.bin", true, false, root);
            Tensor input_1_0({4, 6}, ACL_INT32, 96, "target.bin", true, false, root);
            Tensor output_0_0({1}, ACL_FLOAT, 4, "y.bin", false, false, root);
            Tensor output_1_0({4, 6}, ACL_INT32, 96, "is_target.bin", false, false, root);
            char attr_0[] = "mean";
            uint64_t size = 0;
            aclOpExecutor* executor = nullptr;
            check(aclnnMultilabelMarginLossGetWorkspaceSize(
                input_0_0.tensor,
                input_1_0.tensor,
                attr_0,
                output_0_0.tensor,
                output_1_0.tensor,
                &size,
                &executor));
            void* workspace = nullptr;
            if (size)
                check(aclrtMalloc(&workspace, size, ACL_MEM_MALLOC_HUGE_FIRST));
            check(aclnnMultilabelMarginLoss(workspace, size, executor, stream));
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
