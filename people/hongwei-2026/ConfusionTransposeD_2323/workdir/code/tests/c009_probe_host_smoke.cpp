#include <cstddef>
#include <cstdint>
#include <vector>

namespace gert {
struct ContinuousVector {
    std::vector<int64_t> values;
    size_t GetSize() const { return values.size(); }
    const void* GetData() const { return values.data(); }
};
}

#include "../op_host/confusion_transpose_d_probe.h"

struct Shape {
    std::vector<int64_t> values;
    size_t GetDimNum() const { return values.size(); }
    int64_t GetDim(size_t i) const { return values[i]; }
};
struct Attrs {
    gert::ContinuousVector perm{{1, 0}};
    gert::ContinuousVector shape{{3, 2}};
    bool tf = true;
    template <typename T> const T* GetAttrPointer(size_t index) const
    {
        if (index == 0) return reinterpret_cast<const T*>(&perm);
        if (index == 1) return reinterpret_cast<const T*>(&shape);
        return reinterpret_cast<const T*>(&tf);
    }
};

int main()
{
    Shape input{{2, 3}};
    Shape output{{3, 2}};
    Attrs attrs;
    {
        ctd_probe::Event probe("infer");
        probe.Capture(&input, &attrs, -1);
        probe.Output(&output);
        probe.Success();
    }
    {
        ctd_probe::Event probe("tiling");
        probe.Capture(&input, &attrs, 0);
        probe.OutputStorage(&output);
        probe.Limits(192 * 1024, 6, 4);
        probe.Stage("tensor_too_large_for_ub");
    }
    return 0;
}
