#include "sample_support.h"
#include "aclnn_quant_update_scatter.h"

int main(int argc, char** argv) {
    try {
        std::string root = argc > 1 ? argv[1] : "../samples/build";
        int device = argc > 2 ? std::stoi(argv[2]) : 0;
        QUS_ACL_CHECK(aclInit(nullptr));
        QUS_ACL_CHECK(aclrtSetDevice(device));
        aclrtStream stream;
        QUS_ACL_CHECK(aclrtCreateStream(&stream));
        {
            Tensor input_0_0({1, 2, 32}, ACL_INT8, 64, "var.bin", true, false, root);
            Tensor input_1_0({1}, ACL_INT32, 4, "indices.bin", true, false, root);
            Tensor input_2_0({1, 1, 32}, ACL_FLOAT16, 64, "updates.bin", true, false, root);
            Tensor input_3_0({1, 1, 32}, ACL_FLOAT, 128, "quant_scales.bin", true, false, root);
            Tensor input_4_0({1, 1, 32}, ACL_INT32, 128, "quant_zero_points.bin", true, false, root);
            Tensor output_0_0({1, 2, 32}, ACL_INT8, 64, "var_out.bin", false, false, root);
            char attr_0[] = "update";
            int64_t attr_1 = -2LL;
            int64_t attr_2 = -1LL;
            bool attr_3 = false;
            uint64_t size = 0;
            aclOpExecutor* executor = nullptr;
            QUS_ACL_CHECK(aclnnQuantUpdateScatterGetWorkspaceSize(
                input_0_0.tensor,
                input_1_0.tensor,
                input_2_0.tensor,
                input_3_0.tensor,
                input_4_0.tensor,
                attr_0,
                attr_1,
                attr_2,
                attr_3,
                output_0_0.tensor,
                &size,
                &executor));
            void* workspace = nullptr;
            if (size)
                QUS_ACL_CHECK(aclrtMalloc(&workspace, size, ACL_MEM_MALLOC_HUGE_FIRST));
            QUS_ACL_CHECK(aclnnQuantUpdateScatter(workspace, size, executor, stream));
            QUS_ACL_CHECK(aclrtSynchronizeStream(stream));
            output_0_0.save(root);
            if (workspace)
                QUS_ACL_CHECK(aclrtFree(workspace));
        }
        QUS_ACL_CHECK(aclrtDestroyStream(stream));
        QUS_ACL_CHECK(aclrtResetDevice(device));
        QUS_ACL_CHECK(aclFinalize());
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 2;
    }
}
