#include "sample_support.h"
#include "aclnn_layer_norm_v3.h"

int main(int argc, char** argv) {
    try {
        std::string root = argc > 1 ? argv[1] : "../samples/build";
        int device = argc > 2 ? std::stoi(argv[2]) : 0;
        check(aclInit(nullptr));
        check(aclrtSetDevice(device));
        aclrtStream stream;
        check(aclrtCreateStream(&stream));
        {
            Tensor input_0_0({4, 6, 8}, ACL_FLOAT, 768, "x.bin", true, false, root);
            Tensor input_1_0({6, 8}, ACL_FLOAT, 192, "gamma.bin", true, false, root);
            Tensor input_2_0({6, 8}, ACL_FLOAT, 192, "beta.bin", true, false, root);
            Tensor output_0_0({4, 6, 8}, ACL_FLOAT, 768, "y.bin", false, false, root);
            Tensor output_1_0({4, 1, 1}, ACL_FLOAT, 16, "mean.bin", false, false, root);
            Tensor output_2_0({4, 1, 1}, ACL_FLOAT, 16, "rstd.bin", false, false, root);
            int64_t attr_0 = 1LL;
            int64_t attr_1 = 1LL;
            double attr_2 = static_cast<double>(1e-05);
            uint64_t size = 0;
            aclOpExecutor* executor = nullptr;
            check(aclnnLayerNormV3GetWorkspaceSize(
                input_0_0.tensor,
                input_1_0.tensor,
                input_2_0.tensor,
                attr_0,
                attr_1,
                attr_2,
                output_0_0.tensor,
                output_1_0.tensor,
                output_2_0.tensor,
                &size,
                &executor));
            void* workspace = nullptr;
            if (size)
                check(aclrtMalloc(&workspace, size, ACL_MEM_MALLOC_HUGE_FIRST));
            check(aclnnLayerNormV3(workspace, size, executor, stream));
            check(aclrtSynchronizeStream(stream));
            output_0_0.save(root);
            output_1_0.save(root);
            output_2_0.save(root);
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
