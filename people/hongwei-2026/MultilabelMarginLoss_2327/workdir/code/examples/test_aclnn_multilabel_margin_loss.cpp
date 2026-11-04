#include "sample_support.h"
#include "aclnn_multilabel_margin_loss.h"

int main(int argc, char** argv) {
    try {
        std::string root = argc > 1 ? argv[1] : "../samples/build";
        int device = argc > 2 ? std::stoi(argv[2]) : 0;
        int64_t n = argc > 3 ? std::stoll(argv[3]) : 4;
        int64_t c = argc > 4 ? std::stoll(argv[4]) : 6;
        int64_t k = argc > 5 ? std::stoll(argv[5]) : 6;
        std::string dtype = argc > 6 ? argv[6] : "float32";
        std::string reduction = argc > 7 ? argv[7] : "mean";
        if (n <= 0 || c <= 0 || k <= 0 ||
            (dtype != "float32" && dtype != "float16" && dtype != "bfloat16") ||
            (reduction != "none" && reduction != "mean" && reduction != "sum")) {
            throw std::runtime_error("invalid test shape/dtype/reduction");
        }
        aclDataType xType = dtype == "float32" ? ACL_FLOAT :
            (dtype == "float16" ? ACL_FLOAT16 : ACL_BF16);
        size_t elementBytes = dtype == "float32" ? 4 : 2;
        check(aclInit(nullptr));
        check(aclrtSetDevice(device));
        aclrtStream stream;
        check(aclrtCreateStream(&stream));
        {
            Tensor input_0_0({n, c}, xType, n * c * elementBytes, "x.bin", true, false, root);
            Tensor input_1_0({n, k}, ACL_INT32, n * k * sizeof(int32_t), "target.bin", true, false, root);
            int64_t yElements = reduction == "none" ? n : 1;
            Tensor output_0_0({yElements}, xType, yElements * elementBytes, "y.bin", false, false, root);
            Tensor output_1_0({n, c}, ACL_INT32, n * c * sizeof(int32_t), "is_target.bin", false, false, root);
            std::vector<char> attr_0(reduction.begin(), reduction.end());
            attr_0.push_back('\0');
            uint64_t size = 0;
            aclOpExecutor* executor = nullptr;
            check(aclnnMultilabelMarginLossGetWorkspaceSize(
                input_0_0.tensor,
                input_1_0.tensor,
                attr_0.data(),
                output_0_0.tensor,
                output_1_0.tensor,
                &size,
                &executor));
            std::cout << "workspace_bytes=" << size << std::endl;
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
