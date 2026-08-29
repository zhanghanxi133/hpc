#include "main.h"    
#include <cstdio>    
#include <vector>    
#include <string>    
#include <iostream> 

// 读取整数（用于读取 N）
inline int fast_read_int() {
    int x = 0;
    int c = getchar_unlocked();
    bool neg = false;
    while (c <= ' ') c = getchar_unlocked();
    if (c == '-') {
        neg = true;
        c = getchar_unlocked();
    }
    while (c >= '0' && c <= '9') {
        x = x * 10 + (c - '0');
        c = getchar_unlocked();
    }
    return neg ? -x : x;
}

// 读取浮点数
inline float fast_read_float() {
    double val = 0.0; // 使用 double 保证中间计算的精度
    int c = getchar_unlocked();
    bool neg = false;

    // 跳过空白字符
    while (c <= ' ') c = getchar_unlocked();

    // 处理负号
    if (c == '-') {
        neg = true;
        c = getchar_unlocked();
    }

    // 读取整数部分
    while (c >= '0' && c <= '9') {
        val = val * 10.0 + (c - '0');
        c = getchar_unlocked();
    }

    // 读取小数部分
    if (c == '.') {
        c = getchar_unlocked();
        double factor = 0.1;
        while (c >= '0' && c <= '9') {
            val = val + (c - '0') * factor;
            factor *= 0.1;
            c = getchar_unlocked();
        }
    }
    // 注意：这个简化版本不支持科学计数法 (e.g., 1.23e4)
    
    return neg ? -static_cast<float>(val) : static_cast<float>(val);
}


// 批量输出
static const size_t OUTBUF_SIZE = 1 << 26;  // 64 MB 缓冲
static char outbuf[OUTBUF_SIZE];
static size_t outidx = 0;

// 将一个浮点数写入缓冲
inline void buffer_write_float(float x) {
    // 确保缓冲有足够空间
    if (outidx > OUTBUF_SIZE - 32) {
        fwrite(outbuf, 1, outidx, stdout);
        outidx = 0;
    }
    // 使用 sprintf 直接写入到缓冲中，并更新索引"
    outidx += sprintf(outbuf + outidx, "%.6g ", x);
}

// 刷新缓冲，把剩余内容写出
inline void buffer_flush() {
    if (outidx > 0) {
        fwrite(outbuf, 1, outidx, stdout);
        outidx = 0;
    }
}

// 声明 solve 函数，因为它在 main 之前被调用
void solve(const float* input, float* output, int N);

int main(int argc, char* argv[]) {
    // 1. 参数检查
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }

    // 2. 绑定输入文件 (使用 C-style freopen, 因为 fast_read* 依赖于 stdin)
    if (!freopen(argv[1], "r", stdin)) {
        std::cerr << "file open error: " << argv[1] << std::endl;
        return 1;
    }

    // 读取数据
    int N = fast_read_int();
    std::vector<float> input(N), output(N);
    for (int i = 0; i < N; ++i) {
        input[i] = fast_read_float();
    }

    solve(input.data(), output.data(), N);

    // 输出结果 
    for (int i = 0; i < N; ++i) {
        buffer_write_float(output[i]);
    }

    if (N > 0 && outidx > 0) {
        outbuf[outidx - 1] = '\n'; // 将最后一个空格替换为换行符
    } else if (N == 0) {
        outbuf[outidx++] = '\n';
    }
    
    buffer_flush(); // 把缓冲里的内容一次性写出

    return 0;
}