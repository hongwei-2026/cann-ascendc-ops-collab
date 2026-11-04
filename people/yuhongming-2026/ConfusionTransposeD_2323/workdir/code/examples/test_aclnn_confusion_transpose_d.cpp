#include "sample_support.h"
#include "aclnn_confusion_transpose_d.h"

int main(int argc, char** argv) {
    try {
        std::string root = argc > 1 ? argv[1] : "../samples/build";
        int device = argc > 2 ? std::stoi(argv[2]) : 0;
        check(aclInit(nullptr));
        check(aclrtSetDevice(device));
        aclrtStream stream;
        check(aclrtCreateStream(&stream));
        {
            Tensor input_0_0({2, 3, 4, 5}, ACL_FLOAT, 480, "x.bin", true, false, root);
            Tensor output_0_0({2, 4, 3, 5}, ACL_FLOAT, 480, "y.bin", false, false, root);
            std::array<int64_t,4> attr_0_data{{0LL,2LL,1LL,3LL}};
            auto* attr_0 = aclCreateIntArray(attr_0_data.data(), 4);
            if(!attr_0)throw std::runtime_error("attribute creation failed");
            std::array<int64_t,4> attr_1_data{{2LL,4LL,3LL,5LL}};
            auto* attr_1 = aclCreateIntArray(attr_1_data.data(), 4);
            if(!attr_1)throw std::runtime_error("attribute creation failed");
            bool attr_2 = true;
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
            void* workspace = nullptr;
            if (size)
                check(aclrtMalloc(&workspace, size, ACL_MEM_MALLOC_HUGE_FIRST));
            check(aclnnConfusionTransposeD(workspace, size, executor, stream));
            check(aclrtSynchronizeStream(stream));
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
