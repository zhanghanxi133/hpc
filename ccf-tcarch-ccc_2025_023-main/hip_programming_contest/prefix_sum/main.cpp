#include "main.h"
#include <cstdio>
#include <vector>
#include <climits>   // for INT_MIN

// ------- 快速输入 (基于 getchar_unlocked) -------
inline int fast_read_int() {
    int x = 0;
    int c = getchar_unlocked();
    bool neg = false;

    // 跳过空白字符
    while (c <= ' ') c = getchar_unlocked();

    // 处理负号
    if (c == '-') {
        neg = true;
        c = getchar_unlocked();
    }

    // 读取数字
    while (c >= '0' && c <= '9') {
        x = x * 10 + (c - '0');
        c = getchar_unlocked();
    }
    return neg ? -x : x;
}

// ------- 批量输出版本 (基于 fwrite 大缓冲) -------
static const size_t OUTBUF_SIZE = 1 << 26;  // 64 MB 缓冲，够大但不会爆内存
static char outbuf[OUTBUF_SIZE];
static size_t outidx = 0;

// 将一个整数写入缓冲
inline void buffer_write_int(int x) {
    if (x == 0) {
        outbuf[outidx++] = '0';
        outbuf[outidx++] = ' ';
        return;
    }

    if (x < 0) {
        if (x == INT_MIN) {
            // 特殊处理 INT_MIN（-2147483648）
            const char *s = "-2147483648 ";
            for (const char* p = s; *p; ++p) outbuf[outidx++] = *p;
            return;
        }
        outbuf[outidx++] = '-';
        x = -x;
    }

    // 先写到临时缓冲
    char tmp[20];
    int len = 0;
    while (x > 0) {
        tmp[len++] = '0' + (x % 10);
        x /= 10;
    }
    // 倒序拷贝到 outbuf
    for (int i = len - 1; i >= 0; --i) {
        outbuf[outidx++] = tmp[i];
    }
    outbuf[outidx++] = ' ';

    // 如果缓冲快满了，就先写出去
    if (outidx > OUTBUF_SIZE - 32) {
        fwrite(outbuf, 1, outidx, stdout);
        outidx = 0;
    }
}

// 刷新缓冲，把最后没写的数据写出去
inline void buffer_flush() {
    if (outidx > 0) {
        fwrite(outbuf, 1, outidx, stdout);
        outidx = 0;
    }
}

int main(int argc, char* argv[]) {
    // 检查参数
    if (argc != 2) {
        fprintf(stderr, "usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    // 绑定输入文件
    if (!freopen(argv[1], "r", stdin)) {
        fprintf(stderr, "file open error: %s\n", argv[1]);
        return 1;
    }

    // 读取数据
    int N = fast_read_int();
    std::vector<int> input(N), output(N);
    for (int i = 0; i < N; ++i) {
        input[i] = fast_read_int();
    }

    // 核心 GPU 求解
    solve(input.data(), output.data(), N);

    // 输出结果（放入缓冲）
    for (int i = 0; i < N; ++i) {
        buffer_write_int(output[i]);
    }
    outbuf[outidx++] = '\n';
    buffer_flush(); // 把缓冲里的内容一次性写出

    return 0;
}