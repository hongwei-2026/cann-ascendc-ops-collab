#ifndef CONFUSION_TRANSPOSE_D_PROBE_H
#define CONFUSION_TRANSPOSE_D_PROBE_H

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <sstream>
#include <string>
#include <unistd.h>

namespace ctd_probe {

inline uint64_t NextId()
{
    static std::atomic<uint64_t> sequence{0};
    return ++sequence;
}

template <typename Shape>
void WriteShape(std::ostream& out, const Shape* shape)
{
    if (shape == nullptr) {
        out << "null";
        return;
    }
    out << '[';
    for (size_t i = 0; i < shape->GetDimNum(); ++i) {
        if (i != 0) out << ',';
        out << shape->GetDim(i);
    }
    out << ']';
}

template <typename Vector>
void WriteVector(std::ostream& out, const Vector* vec)
{
    if (vec == nullptr) {
        out << "null";
        return;
    }
    const auto* data = reinterpret_cast<const int64_t*>(vec->GetData());
    if (data == nullptr && vec->GetSize() != 0) {
        out << "null";
        return;
    }
    out << '[';
    for (size_t i = 0; i < vec->GetSize(); ++i) {
        if (i != 0) out << ',';
        out << data[i];
    }
    out << ']';
}

template <typename Shape>
void WriteNumel(std::ostream& out, const Shape* shape)
{
    if (shape == nullptr) { out << "null"; return; }
    int64_t value = 1;
    for (size_t i = 0; i < shape->GetDimNum(); ++i) {
        const int64_t dim = shape->GetDim(i);
        if (dim < 0 || (dim != 0 && value > std::numeric_limits<int64_t>::max() / dim)) {
            out << "null";
            return;
        }
        value *= dim;
    }
    out << value;
}

class Event {
public:
    explicit Event(const char* phase) : phase_(phase), id_(NextId()) {}
    ~Event() { Emit("exit"); }

    template <typename Shape, typename Attrs>
    void Capture(const Shape* input, const Attrs* attrs, int dtype)
    {
        std::ostringstream out;
        out << ",\"dtype_code\":";
        if (dtype < 0) out << "null"; else out << dtype;
        out << ",\"in_shape\":";
        WriteShape(out, input);
        out << ",\"numel\":";
        WriteNumel(out, input);
        out << ",\"perm\":";
        const auto* perm = attrs ? attrs->template GetAttrPointer<gert::ContinuousVector>(0) : nullptr;
        const auto* shape = attrs ? attrs->template GetAttrPointer<gert::ContinuousVector>(1) : nullptr;
        const bool* tf = attrs ? attrs->template GetAttrPointer<bool>(2) : nullptr;
        WriteVector(out, perm);
        out << ",\"attr_shape\":";
        WriteVector(out, shape);
        out << ",\"transpose_first\":";
        if (tf == nullptr) out << "null"; else out << (*tf ? "true" : "false");
        fields_ = out.str();
        Emit("enter");
    }

    template <typename Shape>
    void Output(const Shape* shape)
    {
        std::ostringstream out;
        out << ",\"out_shape\":";
        WriteShape(out, shape);
        extra_ += out.str();
    }

    void Limits(uint64_t ub, int64_t numel, int64_t typeSize)
    {
        std::ostringstream out;
        out << ",\"ub_bytes\":" << ub << ",\"element_bytes\":" << typeSize
            << ",\"input_bytes\":";
        if (numel >= 0 && typeSize > 0 && numel <= std::numeric_limits<int64_t>::max() / typeSize) {
            out << numel * typeSize;
        } else {
            out << "null";
        }
        extra_ += out.str();
    }

    void Tiling(int64_t span, int64_t cores, int64_t block, int64_t tile)
    {
        std::ostringstream out;
        out << ",\"copy_span\":" << span << ",\"path_hint\":\""
            << (span > 1 ? "contiguous_row" : "scalar") << "\",\"block_dim\":" << cores
            << ",\"block_factor\":" << block << ",\"ub_factor\":" << tile;
        extra_ += out.str();
    }

    void Stage(const char* stage) { stage_ = stage; }
    void Success() { success_ = true; stage_ = "complete"; }

private:
    void Emit(const char* event) const
    {
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        std::ostringstream out;
        out << "CTD_PROBE {\"pid\":" << static_cast<long>(getpid()) << ",\"seq\":" << id_
            << ",\"time_unix_us\":" << us << ",\"phase\":\"" << phase_ << "\",\"event\":\""
            << event << "\",\"stage\":\"" << stage_ << "\",\"host_ok\":";
        if (event[1] == 'n') out << "null"; else out << (success_ ? "true" : "false");
        out << fields_ << extra_ << '}';
        const auto line = out.str();
        std::fprintf(stderr, "%s\n", line.c_str());
        std::fflush(stderr);
    }
    const char* phase_;
    uint64_t id_;
    const char* stage_ = "entry";
    bool success_ = false;
    std::string fields_;
    std::string extra_;
};

} // namespace ctd_probe
#endif
