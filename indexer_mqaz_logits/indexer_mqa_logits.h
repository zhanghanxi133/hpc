#pragma once
// INDEXER_SME_BF16_VERSION 2026-08-09 test33_n1_weight_tbl_fmla

#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <arm_bf16.h>
#include <arm_sve.h>
#include <arm_sme.h>
#include <omp.h>

#if defined(INDEXER_KUPL)
#include "kupl.h"
#endif
#include "ref_mqa_logits.h"
#include "allocator.h"
#include "Tensor.h"
#include "utils.h"

namespace indexer_mqa_detail {

constexpr int64_t kMaxNextN = 2;
constexpr int64_t kHeads = 64;
constexpr int64_t kDim = 128;
constexpr int64_t kBlockSize = 64;
constexpr int64_t kDimPairs = kDim / 2;
constexpr int64_t kPhaseTokens = 32;
constexpr int64_t kFourHeadPhaseTokens = 16;
constexpr int kFixedThreads = 32;
constexpr int64_t kMaxBatchSize = 128;
constexpr int64_t kFullTaskPanel = 128;

template <int64_t LocalNextN>
struct alignas(64) QKScratchColor64 {
    static constexpr int64_t kLocalPhaseTokens =
        LocalNextN == 1 ? kFourHeadPhaseTokens : kPhaseTokens;
    alignas(64) bfloat16_t
        q_pair[LocalNextN][kDimPairs][kHeads * 2];
    uint8_t color_pad[64];
    alignas(64) uint32_t k_phase[kDimPairs][kLocalPhaseTokens];
};

static_assert(offsetof(QKScratchColor64<1>, q_pair) == 0);
static_assert(offsetof(QKScratchColor64<1>, k_phase) == 16448);
static_assert(offsetof(QKScratchColor64<2>, q_pair) == 0);
static_assert(offsetof(QKScratchColor64<2>, k_phase) == 32832);
static_assert(sizeof(QKScratchColor64<1>) == 20544);
static_assert(sizeof(QKScratchColor64<2>) == 41024);

struct FullBlockTask {
    const bfloat16_t *kv_block;
    int64_t token_base;
};

struct IndexerTaskArgs {
    const bfloat16_t *__restrict q_data;
    const bfloat16_t *__restrict kv_data;
    const int64_t *__restrict block_table_data;
    const int64_t *__restrict context_len_data;
    const float *__restrict weight_data;
    float *__restrict out_data;
    const int64_t *__restrict block_prefix;
    int64_t batch_size;
    int64_t next_n;
    int64_t max_num_blocks;
    int64_t max_model_len;
    int64_t total_blocks;
};

template <bool WaveMajor>
__attribute__((always_inline)) inline void pack_q_pairs(
    const bfloat16_t *__restrict q_src,
    bfloat16_t *__restrict q_dst,
    svbool_t pg_all_f32,
    svbool_t pg_all_u64,
    svint64_t gather_offsets_u64)
{
    constexpr int64_t kPackedPairStride =
        WaveMajor ? kHeads : kHeads * 2;
    constexpr int64_t kWaveStride = kDimPairs * kHeads;

    for (int64_t p = 0; p < kDimPairs; p += 2) {
        for (int64_t h = 0; h < kHeads; h += 16) {
            const uint64_t *__restrict q_heads_lo =
                reinterpret_cast<const uint64_t *>(
                    q_src + h * kDim + p * 2);
            const uint64_t *__restrict q_heads_hi =
                reinterpret_cast<const uint64_t *>(
                    q_src + (h + 8) * kDim + p * 2);
            const svuint32_t packed_lo = svreinterpret_u32_u64(
                svld1_gather_s64offset_u64(
                    pg_all_u64, q_heads_lo, gather_offsets_u64));
            const svuint32_t packed_hi = svreinterpret_u32_u64(
                svld1_gather_s64offset_u64(
                    pg_all_u64, q_heads_hi, gather_offsets_u64));

            int64_t dst_offset;
            if constexpr (WaveMajor) {
                dst_offset = (h / 32) * kWaveStride +
                    p * kPackedPairStride + (h % 32) * 2;
            } else {
                dst_offset = p * kPackedPairStride + h * 2;
            }
            svst1_u32(pg_all_f32,
                reinterpret_cast<uint32_t *>(q_dst + dst_offset),
                svuzp1_u32(packed_lo, packed_hi));
            svst1_u32(pg_all_f32,
                reinterpret_cast<uint32_t *>(
                    q_dst + dst_offset + kPackedPairStride),
                svuzp2_u32(packed_lo, packed_hi));
        }
    }
}

inline void prefetch_bf16_phase_l2(
    const bfloat16_t *__restrict block,
    int64_t phase_token_base) __arm_streaming;
    
__arm_locally_streaming
__arm_new("za")
__attribute__((noinline))
void sme_bf16_block_reduce(const bfloat16_t *__restrict q_pair,
                           const uint32_t *__restrict k_pairs,
                           const float *__restrict weights,
                           float *__restrict output,
                           int64_t tokens_this_n)
{
    // The competition target has 512-bit streaming vectors: 16 FP32 lanes
    // and 32 BF16 lanes.  Pairing two token tiles with two head tiles lets
    // each packed Q/K vector feed two ZA tiles.
    constexpr uint64_t kTileRows = 16;
    const svbool_t pg_heads_bf16 =
        svwhilelt_b16_u64(0, 2 * kTileRows);
    const svfloat32_t zero = svdup_n_f32(0.0f);

    for (uint64_t token_base = 0;
         token_base < static_cast<uint64_t>(tokens_this_n);
         token_base += 2 * kTileRows) {
        const uint64_t remaining =
            static_cast<uint64_t>(tokens_this_n) - token_base;
        const uint64_t tokens0 =
            std::min<uint64_t>(kTileRows, remaining);
        const bool has_second_tile = remaining > kTileRows;
        const uint64_t tokens1 = has_second_tile
            ? std::min<uint64_t>(kTileRows, remaining - kTileRows)
            : 0;

        const svbool_t pg0_f32 = svwhilelt_b32_u64(0, tokens0);
        const svbool_t pg0_bf16 = svwhilelt_b16_u64(0, tokens0 * 2);
        const svbool_t pg1_f32 = svwhilelt_b32_u64(0, tokens1);
        const svbool_t pg1_bf16 = svwhilelt_b16_u64(0, tokens1 * 2);

        svfloat32_t acc00 = zero;
        svfloat32_t acc01 = zero;
        svfloat32_t acc10 = zero;
        svfloat32_t acc11 = zero;

        // Head tiles [0,16) and [16,32): ZA0/1 hold the first/second
        // token tile for the first head tile; ZA2/3 do the same for the
        // second head tile.
        svzero_za();
        const uint32_t *__restrict k_ptr = k_pairs + token_base;
        for (int64_t p = 0; p < kDimPairs; ++p) {
            const bfloat16_t *__restrict q_base =
                q_pair + p * kHeads * 2;
            const svbfloat16_t q0 =
                svld1_bf16(pg_heads_bf16, q_base);
            const svbfloat16_t q1 =
                svld1_bf16(pg_heads_bf16, q_base + 16 * 2);
            const svuint32_t k0_u32 = svld1_u32(pg0_f32, k_ptr);
            const svbfloat16_t k0 = svreinterpret_bf16_u32(k0_u32);

            svmopa_za32_bf16_m(0, pg_heads_bf16, pg0_bf16, q0, k0);
            svmopa_za32_bf16_m(2, pg_heads_bf16, pg0_bf16, q1, k0);

            if (has_second_tile) {
                const svuint32_t k1_u32 =
                    svld1_u32(pg1_f32, k_ptr + kTileRows);
                const svbfloat16_t k1 = svreinterpret_bf16_u32(k1_u32);
                svmopa_za32_bf16_m(1, pg_heads_bf16, pg1_bf16, q0, k1);
                svmopa_za32_bf16_m(3, pg_heads_bf16, pg1_bf16, q1, k1);
            }
            k_ptr += kBlockSize;
        }

        {
            // Active lanes are overwritten by each ZA read; inactive lanes
            // are never consumed or stored.  Keep one merge register per ZA
            // tile instead of copying the zero vector for every head row.
            svfloat32_t score00 = zero;
            svfloat32_t score01 = zero;
            svfloat32_t score10 = zero;
            svfloat32_t score11 = zero;

            for (uint32_t r = 0; r < kTileRows; ++r) {
                const svfloat32_t w0 = svdup_f32(weights[r]);
                const svfloat32_t w1 = svdup_f32(weights[kTileRows + r]);

                score00 = svread_hor_za32_f32_m(
                    score00, pg0_f32, 0, r);
                const svfloat32_t relu00 =
                    svmax_n_f32_m(pg0_f32, score00, 0.0f);
                acc00 = svmla_f32_m(pg0_f32, acc00, relu00, w0);

                if (has_second_tile) {
                    score01 = svread_hor_za32_f32_m(
                        score01, pg1_f32, 1, r);
                    const svfloat32_t relu01 =
                        svmax_n_f32_m(pg1_f32, score01, 0.0f);
                    acc01 = svmla_f32_m(pg1_f32, acc01, relu01, w0);
                }

                score10 = svread_hor_za32_f32_m(
                    score10, pg0_f32, 2, r);
                const svfloat32_t relu10 =
                    svmax_n_f32_m(pg0_f32, score10, 0.0f);
                acc10 = svmla_f32_m(pg0_f32, acc10, relu10, w1);

                if (has_second_tile) {
                    score11 = svread_hor_za32_f32_m(
                        score11, pg1_f32, 3, r);
                    const svfloat32_t relu11 =
                        svmax_n_f32_m(pg1_f32, score11, 0.0f);
                    acc11 = svmla_f32_m(pg1_f32, acc11, relu11, w1);
                }
            }
        }

        // Head tiles [32,48) and [48,64), reusing the same two token tiles.
        svfloat32_t acc20 = zero;
        svfloat32_t acc21 = zero;
        svfloat32_t acc30 = zero;
        svfloat32_t acc31 = zero;
        svzero_za();
        const uint32_t *__restrict k_ptr2 = k_pairs + token_base;
        for (int64_t p = 0; p < kDimPairs; ++p) {
            const bfloat16_t *__restrict q_base =
                q_pair + p * kHeads * 2;
            const svbfloat16_t q2 =
                svld1_bf16(pg_heads_bf16, q_base + 32 * 2);
            const svbfloat16_t q3 =
                svld1_bf16(pg_heads_bf16, q_base + 48 * 2);
            const svuint32_t k0_u32 = svld1_u32(pg0_f32, k_ptr2);
            const svbfloat16_t k0 = svreinterpret_bf16_u32(k0_u32);

            svmopa_za32_bf16_m(0, pg_heads_bf16, pg0_bf16, q2, k0);
            svmopa_za32_bf16_m(2, pg_heads_bf16, pg0_bf16, q3, k0);

            if (has_second_tile) {
                const svuint32_t k1_u32 =
                    svld1_u32(pg1_f32, k_ptr2 + kTileRows);
                const svbfloat16_t k1 = svreinterpret_bf16_u32(k1_u32);
                svmopa_za32_bf16_m(1, pg_heads_bf16, pg1_bf16, q2, k1);
                svmopa_za32_bf16_m(3, pg_heads_bf16, pg1_bf16, q3, k1);
            }
            k_ptr2 += kBlockSize;
        }

        {
            svfloat32_t score20 = zero;
            svfloat32_t score21 = zero;
            svfloat32_t score30 = zero;
            svfloat32_t score31 = zero;

            for (uint32_t r = 0; r < kTileRows; ++r) {
                const svfloat32_t w2 =
                    svdup_f32(weights[2 * kTileRows + r]);
                const svfloat32_t w3 =
                    svdup_f32(weights[3 * kTileRows + r]);

                score20 = svread_hor_za32_f32_m(
                    score20, pg0_f32, 0, r);
                const svfloat32_t relu20 =
                    svmax_n_f32_m(pg0_f32, score20, 0.0f);
                acc20 = svmla_f32_m(pg0_f32, acc20, relu20, w2);

                if (has_second_tile) {
                    score21 = svread_hor_za32_f32_m(
                        score21, pg1_f32, 1, r);
                    const svfloat32_t relu21 =
                        svmax_n_f32_m(pg1_f32, score21, 0.0f);
                    acc21 = svmla_f32_m(pg1_f32, acc21, relu21, w2);
                }

                score30 = svread_hor_za32_f32_m(
                    score30, pg0_f32, 2, r);
                const svfloat32_t relu30 =
                    svmax_n_f32_m(pg0_f32, score30, 0.0f);
                acc30 = svmla_f32_m(pg0_f32, acc30, relu30, w3);

                if (has_second_tile) {
                    score31 = svread_hor_za32_f32_m(
                        score31, pg1_f32, 3, r);
                    const svfloat32_t relu31 =
                        svmax_n_f32_m(pg1_f32, score31, 0.0f);
                    acc31 = svmla_f32_m(pg1_f32, acc31, relu31, w3);
                }
            }
        }

        svfloat32_t sum0 = svadd_f32_m(pg0_f32, acc00, acc10);
        svfloat32_t sum2 = svadd_f32_m(pg0_f32, acc20, acc30);
        const svfloat32_t result0 = svadd_f32_m(pg0_f32, sum0, sum2);
        svst1_f32(pg0_f32, output + token_base, result0);

        if (has_second_tile) {
            svfloat32_t sum1 = svadd_f32_m(pg1_f32, acc01, acc11);
            svfloat32_t sum3 = svadd_f32_m(pg1_f32, acc21, acc31);
            const svfloat32_t result1 = svadd_f32_m(pg1_f32, sum1, sum3);
            svst1_f32(pg1_f32, output + token_base + kTileRows, result1);
        }
    }
}

// Keep one SME region active for every full-block range owned by a thread.
// Each 32-token phase is packed into an 8 KiB scratch buffer and consumed by
// every query before the same buffer is reused for the next phase.
__arm_locally_streaming
__arm_new("za")
__attribute__((noinline))
void sme_bf16_full_range_pipeline(
    const FullBlockTask *__restrict tasks,
    int64_t block_count,
    uint32_t *__restrict k_phase,
    const bfloat16_t *__restrict q_pair,
    const float *__restrict weights,
    float *__restrict output,
    int64_t output_stride,
    const int64_t *__restrict valid_lens,
    int64_t next_n)
{
    constexpr uint64_t kTokenTile = 16;
    constexpr uint64_t kTileRows = 16;
    const svbool_t pg_all_f32 = svwhilelt_b32_u64(0, kTokenTile);
    const svbool_t pg_heads_bf16 =
        svwhilelt_b16_u64(0, 2 * kTileRows);
    const svfloat32_t zero = svdup_n_f32(0.0f);

    for (int64_t block_pos = 0; block_pos < block_count; ++block_pos) {
        const bfloat16_t *__restrict kv_block = tasks[block_pos].kv_block;
        const int64_t block_token_base = tasks[block_pos].token_base;

        for (int64_t token_phase = 0; token_phase < 2; ++token_phase) {
            const int64_t phase_token_base = token_phase * kPhaseTokens;

            for (uint64_t slab = 0; slab < kPhaseTokens;
                 slab += kTokenTile) {
                const bfloat16_t *__restrict src =
                    kv_block + (phase_token_base + slab) * kDim;

                for (uint32_t row = 0; row < kTokenTile; ++row) {
                    svld1_hor_za32(0, row, pg_all_f32, src + 0 * 32);
                    svld1_hor_za32(1, row, pg_all_f32, src + 1 * 32);
                    svld1_hor_za32(2, row, pg_all_f32, src + 2 * 32);
                    svld1_hor_za32(3, row, pg_all_f32, src + 3 * 32);
                    src += kDim;
                }

                #pragma clang loop unroll(disable)
                for (uint32_t column = 0; column < 16; column += 4) {
                    svst1_ver_za32(
                        0, column + 0, pg_all_f32,
                        k_phase + (0 * 16 + column + 0) * kPhaseTokens + slab);
                    svst1_ver_za32(
                        0, column + 1, pg_all_f32,
                        k_phase + (0 * 16 + column + 1) * kPhaseTokens + slab);
                    svst1_ver_za32(
                        0, column + 2, pg_all_f32,
                        k_phase + (0 * 16 + column + 2) * kPhaseTokens + slab);
                    svst1_ver_za32(
                        0, column + 3, pg_all_f32,
                        k_phase + (0 * 16 + column + 3) * kPhaseTokens + slab);
                }
                #pragma clang loop unroll(disable)
                for (uint32_t column = 0; column < 16; column += 4) {
                    svst1_ver_za32(
                        1, column + 0, pg_all_f32,
                        k_phase + (1 * 16 + column + 0) * kPhaseTokens + slab);
                    svst1_ver_za32(
                        1, column + 1, pg_all_f32,
                        k_phase + (1 * 16 + column + 1) * kPhaseTokens + slab);
                    svst1_ver_za32(
                        1, column + 2, pg_all_f32,
                        k_phase + (1 * 16 + column + 2) * kPhaseTokens + slab);
                    svst1_ver_za32(
                        1, column + 3, pg_all_f32,
                        k_phase + (1 * 16 + column + 3) * kPhaseTokens + slab);
                }
                #pragma clang loop unroll(disable)
                for (uint32_t column = 0; column < 16; column += 4) {
                    svst1_ver_za32(
                        2, column + 0, pg_all_f32,
                        k_phase + (2 * 16 + column + 0) * kPhaseTokens + slab);
                    svst1_ver_za32(
                        2, column + 1, pg_all_f32,
                        k_phase + (2 * 16 + column + 1) * kPhaseTokens + slab);
                    svst1_ver_za32(
                        2, column + 2, pg_all_f32,
                        k_phase + (2 * 16 + column + 2) * kPhaseTokens + slab);
                    svst1_ver_za32(
                        2, column + 3, pg_all_f32,
                        k_phase + (2 * 16 + column + 3) * kPhaseTokens + slab);
                }
                #pragma clang loop unroll(disable)
                for (uint32_t column = 0; column < 16; column += 4) {
                    svst1_ver_za32(
                        3, column + 0, pg_all_f32,
                        k_phase + (3 * 16 + column + 0) * kPhaseTokens + slab);
                    svst1_ver_za32(
                        3, column + 1, pg_all_f32,
                        k_phase + (3 * 16 + column + 1) * kPhaseTokens + slab);
                    svst1_ver_za32(
                        3, column + 2, pg_all_f32,
                        k_phase + (3 * 16 + column + 2) * kPhaseTokens + slab);
                    svst1_ver_za32(
                        3, column + 3, pg_all_f32,
                        k_phase + (3 * 16 + column + 3) * kPhaseTokens + slab);
                }
            }

            if (block_pos + 1 < block_count) {
                prefetch_bf16_phase_l2(
                    tasks[block_pos + 1].kv_block, phase_token_base);
            }

            for (int64_t n = 0; n < next_n; ++n) {
                const int64_t remaining =
                    valid_lens[n] - block_token_base - phase_token_base;
                if (remaining <= 0) {
                    continue;
                }
                const uint64_t phase_tokens =
                    static_cast<uint64_t>(std::min<int64_t>(
                        kPhaseTokens, remaining));
                const uint64_t tokens0 =
                    std::min<uint64_t>(kTileRows, phase_tokens);
                const bool has_second_tile = phase_tokens > kTileRows;
                const uint64_t tokens1 = has_second_tile
                    ? phase_tokens - kTileRows
                    : 0;

                const svbool_t pg0_f32 = svwhilelt_b32_u64(0, tokens0);
                const svbool_t pg0_bf16 =
                    svwhilelt_b16_u64(0, tokens0 * 2);
                const svbool_t pg1_f32 = svwhilelt_b32_u64(0, tokens1);
                const svbool_t pg1_bf16 =
                    svwhilelt_b16_u64(0, tokens1 * 2);
                const bfloat16_t *__restrict q_n =
                    q_pair + n * kDimPairs * kHeads * 2;
                const float *__restrict weights_n = weights + n * kHeads;
                float *__restrict output_n =
                    output + n * output_stride + block_token_base +
                    phase_token_base;

                svfloat32_t acc00 = zero;
                svfloat32_t acc01 = zero;
                svfloat32_t acc10 = zero;
                svfloat32_t acc11 = zero;

                svzero_za();
                const uint32_t *__restrict phase_k_ptr = k_phase;
                for (int64_t p = 0; p < kDimPairs; ++p) {
                    const bfloat16_t *__restrict q_base =
                        q_n + p * kHeads * 2;
                    const svbfloat16_t q0 =
                        svld1_bf16(pg_heads_bf16, q_base);
                    const svbfloat16_t q1 =
                        svld1_bf16(pg_heads_bf16, q_base + 16 * 2);
                    const svuint32_t k0_u32 =
                        svld1_u32(pg0_f32, phase_k_ptr);
                    const svbfloat16_t k0 =
                        svreinterpret_bf16_u32(k0_u32);

                    svmopa_za32_bf16_m(
                        0, pg_heads_bf16, pg0_bf16, q0, k0);
                    svmopa_za32_bf16_m(
                        2, pg_heads_bf16, pg0_bf16, q1, k0);

                    if (has_second_tile) {
                        const svuint32_t k1_u32 =
                            svld1_u32(pg1_f32, phase_k_ptr + kTileRows);
                        const svbfloat16_t k1 =
                            svreinterpret_bf16_u32(k1_u32);
                        svmopa_za32_bf16_m(
                            1, pg_heads_bf16, pg1_bf16, q0, k1);
                        svmopa_za32_bf16_m(
                            3, pg_heads_bf16, pg1_bf16, q1, k1);
                    }
                    phase_k_ptr += kPhaseTokens;
                }

                {
                    svfloat32_t score00 = zero;
                    svfloat32_t score01 = zero;
                    svfloat32_t score10 = zero;
                    svfloat32_t score11 = zero;

                    for (uint32_t r = 0; r < kTileRows; ++r) {
                        const svfloat32_t w0 = svdup_f32(weights_n[r]);
                        const svfloat32_t w1 =
                            svdup_f32(weights_n[kTileRows + r]);

                        score00 = svread_hor_za32_f32_m(
                            score00, pg0_f32, 0, r);
                        const svfloat32_t relu00 =
                            svmax_n_f32_m(pg0_f32, score00, 0.0f);
                        acc00 = svmla_f32_m(
                            pg0_f32, acc00, relu00, w0);

                        if (has_second_tile) {
                            score01 = svread_hor_za32_f32_m(
                                score01, pg1_f32, 1, r);
                            const svfloat32_t relu01 =
                                svmax_n_f32_m(pg1_f32, score01, 0.0f);
                            acc01 = svmla_f32_m(
                                pg1_f32, acc01, relu01, w0);
                        }

                        score10 = svread_hor_za32_f32_m(
                            score10, pg0_f32, 2, r);
                        const svfloat32_t relu10 =
                            svmax_n_f32_m(pg0_f32, score10, 0.0f);
                        acc10 = svmla_f32_m(
                            pg0_f32, acc10, relu10, w1);

                        if (has_second_tile) {
                            score11 = svread_hor_za32_f32_m(
                                score11, pg1_f32, 3, r);
                            const svfloat32_t relu11 =
                                svmax_n_f32_m(pg1_f32, score11, 0.0f);
                            acc11 = svmla_f32_m(
                                pg1_f32, acc11, relu11, w1);
                        }
                    }
                }

                // Collapse the first head wave before loading the second one.
                // This keeps the original reduction tree while reducing live
                // Z state across the following load-heavy dim loop.
                const svfloat32_t first_sum0 =
                    svadd_f32_m(pg0_f32, acc00, acc10);
                const svfloat32_t first_sum1 =
                    svadd_f32_m(pg1_f32, acc01, acc11);

                svfloat32_t acc20 = zero;
                svfloat32_t acc21 = zero;
                svfloat32_t acc30 = zero;
                svfloat32_t acc31 = zero;
                svzero_za();
                const uint32_t *__restrict phase_k_ptr2 = k_phase;
                for (int64_t p = 0; p < kDimPairs; ++p) {
                    const bfloat16_t *__restrict q_base =
                        q_n + p * kHeads * 2;
                    const svbfloat16_t q2 = svld1_bf16(
                        pg_heads_bf16, q_base + 32 * 2);
                    const svbfloat16_t q3 = svld1_bf16(
                        pg_heads_bf16, q_base + 48 * 2);
                    const svuint32_t k0_u32 =
                        svld1_u32(pg0_f32, phase_k_ptr2);
                    const svbfloat16_t k0 =
                        svreinterpret_bf16_u32(k0_u32);

                    svmopa_za32_bf16_m(
                        0, pg_heads_bf16, pg0_bf16, q2, k0);
                    svmopa_za32_bf16_m(
                        2, pg_heads_bf16, pg0_bf16, q3, k0);

                    if (has_second_tile) {
                        const svuint32_t k1_u32 =
                            svld1_u32(pg1_f32, phase_k_ptr2 + kTileRows);
                        const svbfloat16_t k1 =
                            svreinterpret_bf16_u32(k1_u32);
                        svmopa_za32_bf16_m(
                            1, pg_heads_bf16, pg1_bf16, q2, k1);
                        svmopa_za32_bf16_m(
                            3, pg_heads_bf16, pg1_bf16, q3, k1);
                    }
                    phase_k_ptr2 += kPhaseTokens;
                }

                {
                    svfloat32_t score20 = zero;
                    svfloat32_t score21 = zero;
                    svfloat32_t score30 = zero;
                    svfloat32_t score31 = zero;

                    for (uint32_t r = 0; r < kTileRows; ++r) {
                        const svfloat32_t w2 =
                            svdup_f32(weights_n[2 * kTileRows + r]);
                        const svfloat32_t w3 =
                            svdup_f32(weights_n[3 * kTileRows + r]);

                        score20 = svread_hor_za32_f32_m(
                            score20, pg0_f32, 0, r);
                        const svfloat32_t relu20 =
                            svmax_n_f32_m(pg0_f32, score20, 0.0f);
                        acc20 = svmla_f32_m(
                            pg0_f32, acc20, relu20, w2);

                        if (has_second_tile) {
                            score21 = svread_hor_za32_f32_m(
                                score21, pg1_f32, 1, r);
                            const svfloat32_t relu21 =
                                svmax_n_f32_m(pg1_f32, score21, 0.0f);
                            acc21 = svmla_f32_m(
                                pg1_f32, acc21, relu21, w2);
                        }

                        score30 = svread_hor_za32_f32_m(
                            score30, pg0_f32, 2, r);
                        const svfloat32_t relu30 =
                            svmax_n_f32_m(pg0_f32, score30, 0.0f);
                        acc30 = svmla_f32_m(
                            pg0_f32, acc30, relu30, w3);

                        if (has_second_tile) {
                            score31 = svread_hor_za32_f32_m(
                                score31, pg1_f32, 3, r);
                            const svfloat32_t relu31 =
                                svmax_n_f32_m(pg1_f32, score31, 0.0f);
                            acc31 = svmla_f32_m(
                                pg1_f32, acc31, relu31, w3);
                        }
                    }
                }

                const svfloat32_t second_sum0 =
                    svadd_f32_m(pg0_f32, acc20, acc30);
                const svfloat32_t result0 =
                    svadd_f32_m(pg0_f32, first_sum0, second_sum0);
                svst1_f32(pg0_f32, output_n, result0);

                if (has_second_tile) {
                    const svfloat32_t second_sum1 =
                        svadd_f32_m(pg1_f32, acc21, acc31);
                    const svfloat32_t result1 =
                        svadd_f32_m(pg1_f32, first_sum1, second_sum1);
                    svst1_f32(
                        pg1_f32, output_n + kTileRows, result1);
                }
            }
        }
    }
}

// Stage one 8 KiB phase of the next random physical block in private L2.  The
// current phase's score work provides enough distance for these line prefetches
// without pulling the following phase into L1 prematurely.
__attribute__((always_inline)) inline void prefetch_bf16_phase_l2(
    const bfloat16_t *__restrict block,
    int64_t phase_token_base) __arm_streaming
{
    constexpr int64_t kBf16PerCacheLine = 32;
    const int64_t begin = phase_token_base * kDim;
    const int64_t end = begin + kPhaseTokens * kDim;
    for (int64_t offset = begin; offset < end;
         offset += kBf16PerCacheLine) {
        __builtin_prefetch(block + offset, 0, 2);
    }
}

// Keep one SME region active for every n1 block range owned by a thread.  The
// physical cache always contains 64 initialized tokens and the official output
// stride is block aligned, so a partial final block can use the same unmasked
// hot path; the deferred mask overwrites its out-of-context lanes afterwards.
__arm_locally_streaming
__arm_new("za")
__attribute__((noinline))
void sme_bf16_full_range_pipeline_n1(
    const FullBlockTask *__restrict tasks,
    int64_t block_count,
    uint32_t *__restrict k_phase,
    const bfloat16_t *__restrict q_pair,
    const float *__restrict weights,
    float *__restrict output,
    int64_t output_stride,
    int64_t next_n)
{
    constexpr uint64_t kTokenTile = 16;
    constexpr uint64_t kTileRows = 16;
    constexpr int64_t kQWavePairStride = kHeads;
    constexpr int64_t kQWaveStride = kDimPairs * kQWavePairStride;
    const svbool_t pg_all_f32 = svwhilelt_b32_u64(0, kTokenTile);
    const svbool_t pg_heads_bf16 =
        svwhilelt_b16_u64(0, 2 * kTileRows);
    const svfloat32_t zero = svdup_n_f32(0.0f);

    for (int64_t block_pos = 0; block_pos < block_count; ++block_pos) {
        const bfloat16_t *__restrict kv_block = tasks[block_pos].kv_block;
        const int64_t block_token_base = tasks[block_pos].token_base;

        for (int64_t token_phase = 0; token_phase < 2; ++token_phase) {
            const int64_t phase_token_base = token_phase * kPhaseTokens;

            for (uint64_t slab = 0; slab < kPhaseTokens;
                 slab += kTokenTile) {
                const bfloat16_t *__restrict src =
                    kv_block + (phase_token_base + slab) * kDim;

                for (uint32_t row = 0; row < kTokenTile; ++row) {
                    svld1_hor_za32(0, row, pg_all_f32, src + 0 * 32);
                    svld1_hor_za32(1, row, pg_all_f32, src + 1 * 32);
                    svld1_hor_za32(2, row, pg_all_f32, src + 2 * 32);
                    svld1_hor_za32(3, row, pg_all_f32, src + 3 * 32);
                    src += kDim;
                }

                #pragma clang loop unroll(disable)
                for (uint32_t column = 0; column < 16; column += 4) {
                    svst1_ver_za32(
                        0, column + 0, pg_all_f32,
                        k_phase + (0 * 16 + column + 0) * kPhaseTokens + slab);
                    svst1_ver_za32(
                        0, column + 1, pg_all_f32,
                        k_phase + (0 * 16 + column + 1) * kPhaseTokens + slab);
                    svst1_ver_za32(
                        0, column + 2, pg_all_f32,
                        k_phase + (0 * 16 + column + 2) * kPhaseTokens + slab);
                    svst1_ver_za32(
                        0, column + 3, pg_all_f32,
                        k_phase + (0 * 16 + column + 3) * kPhaseTokens + slab);
                }
                #pragma clang loop unroll(disable)
                for (uint32_t column = 0; column < 16; column += 4) {
                    svst1_ver_za32(
                        1, column + 0, pg_all_f32,
                        k_phase + (1 * 16 + column + 0) * kPhaseTokens + slab);
                    svst1_ver_za32(
                        1, column + 1, pg_all_f32,
                        k_phase + (1 * 16 + column + 1) * kPhaseTokens + slab);
                    svst1_ver_za32(
                        1, column + 2, pg_all_f32,
                        k_phase + (1 * 16 + column + 2) * kPhaseTokens + slab);
                    svst1_ver_za32(
                        1, column + 3, pg_all_f32,
                        k_phase + (1 * 16 + column + 3) * kPhaseTokens + slab);
                }
                #pragma clang loop unroll(disable)
                for (uint32_t column = 0; column < 16; column += 4) {
                    svst1_ver_za32(
                        2, column + 0, pg_all_f32,
                        k_phase + (2 * 16 + column + 0) * kPhaseTokens + slab);
                    svst1_ver_za32(
                        2, column + 1, pg_all_f32,
                        k_phase + (2 * 16 + column + 1) * kPhaseTokens + slab);
                    svst1_ver_za32(
                        2, column + 2, pg_all_f32,
                        k_phase + (2 * 16 + column + 2) * kPhaseTokens + slab);
                    svst1_ver_za32(
                        2, column + 3, pg_all_f32,
                        k_phase + (2 * 16 + column + 3) * kPhaseTokens + slab);
                }
                #pragma clang loop unroll(disable)
                for (uint32_t column = 0; column < 16; column += 4) {
                    svst1_ver_za32(
                        3, column + 0, pg_all_f32,
                        k_phase + (3 * 16 + column + 0) * kPhaseTokens + slab);
                    svst1_ver_za32(
                        3, column + 1, pg_all_f32,
                        k_phase + (3 * 16 + column + 1) * kPhaseTokens + slab);
                    svst1_ver_za32(
                        3, column + 2, pg_all_f32,
                        k_phase + (3 * 16 + column + 2) * kPhaseTokens + slab);
                    svst1_ver_za32(
                        3, column + 3, pg_all_f32,
                        k_phase + (3 * 16 + column + 3) * kPhaseTokens + slab);
                }
            }

            if (block_pos + 1 < block_count) {
                prefetch_bf16_phase_l2(
                    tasks[block_pos + 1].kv_block, phase_token_base);
            }

            for (int64_t n = 0; n < next_n; ++n) {
                const bfloat16_t *__restrict q_n =
                    q_pair + n * kDimPairs * kHeads * 2;
                const float *__restrict weights_n = weights + n * kHeads;
                float *__restrict output_n =
                    output + n * output_stride + block_token_base +
                    phase_token_base;
                const svfloat32_t weight0 =
                    svld1_f32(pg_all_f32, weights_n + 0 * kTileRows);
                const svfloat32_t weight1 =
                    svld1_f32(pg_all_f32, weights_n + 1 * kTileRows);
                const svfloat32_t weight2 =
                    svld1_f32(pg_all_f32, weights_n + 2 * kTileRows);
                const svfloat32_t weight3 =
                    svld1_f32(pg_all_f32, weights_n + 3 * kTileRows);

                svfloat32_t acc00 = zero;
                svfloat32_t acc01 = zero;
                svfloat32_t acc10 = zero;
                svfloat32_t acc11 = zero;

                svzero_za();
                const uint32_t *__restrict phase_k_ptr = k_phase;
#pragma unroll 4
                for (int64_t p = 0; p < kDimPairs; ++p) {
                    const bfloat16_t *__restrict q_base =
                        q_n + p * kQWavePairStride;
                    const svuint32_t k0_u32 =
                        svld1_u32(pg_all_f32, phase_k_ptr);
                    const svuint32_t k1_u32 =
                        svld1_u32(pg_all_f32,
                                  phase_k_ptr + kTileRows);
                    const svbfloat16_t k0 =
                        svreinterpret_bf16_u32(k0_u32);
                    const svbfloat16_t k1 =
                        svreinterpret_bf16_u32(k1_u32);
                    const svbfloat16_t q0 =
                        svld1_bf16(pg_heads_bf16, q_base);
                    const svbfloat16_t q1 =
                        svld1_bf16(pg_heads_bf16, q_base + 16 * 2);

                    svmopa_za32_bf16_m(
                        0, pg_heads_bf16, pg_heads_bf16, q0, k0);
                    svmopa_za32_bf16_m(
                        1, pg_heads_bf16, pg_heads_bf16, q0, k1);
                    svmopa_za32_bf16_m(
                        2, pg_heads_bf16, pg_heads_bf16, q1, k0);
                    svmopa_za32_bf16_m(
                        3, pg_heads_bf16, pg_heads_bf16, q1, k1);
                    phase_k_ptr += kPhaseTokens;
                }

                {
#define REDUCE_FULL_01_ROW(R, L, W0, W1)                                  \
                    do {                                                    \
                        svfloat32_t s00 = svread_hor_za32_f32_m(            \
                            zero, pg_all_f32, 0, (R));                      \
                        svfloat32_t s01 = svread_hor_za32_f32_m(            \
                            zero, pg_all_f32, 1, (R));                      \
                        svfloat32_t s10 = svread_hor_za32_f32_m(            \
                            zero, pg_all_f32, 2, (R));                      \
                        svfloat32_t s11 = svread_hor_za32_f32_m(            \
                            zero, pg_all_f32, 3, (R));                      \
                        s00 = svmax_n_f32_m(pg_all_f32, s00, 0.0f);         \
                        s01 = svmax_n_f32_m(pg_all_f32, s01, 0.0f);         \
                        s10 = svmax_n_f32_m(pg_all_f32, s10, 0.0f);         \
                        s11 = svmax_n_f32_m(pg_all_f32, s11, 0.0f);         \
                        const svfloat32_t w0 =                              \
                            svdup_lane_f32((W0), (L));                      \
                        const svfloat32_t w1 =                              \
                            svdup_lane_f32((W1), (L));                      \
                        acc00 = svmla_f32_m(                                \
                            pg_all_f32, acc00, s00, w0);                    \
                        acc01 = svmla_f32_m(                                \
                            pg_all_f32, acc01, s01, w0);                    \
                        acc10 = svmla_f32_m(                                \
                            pg_all_f32, acc10, s10, w1);                    \
                        acc11 = svmla_f32_m(                                \
                            pg_all_f32, acc11, s11, w1);                    \
                    } while (0)
                    svfloat32_t rw0 = weight0;
                    svfloat32_t rw1 = weight1;
                    REDUCE_FULL_01_ROW(0, 0, rw0, rw1);
                    REDUCE_FULL_01_ROW(1, 1, rw0, rw1);
                    REDUCE_FULL_01_ROW(2, 2, rw0, rw1);
                    REDUCE_FULL_01_ROW(3, 3, rw0, rw1);
                    rw0 = svext_f32(rw0, rw0, 4);
                    rw1 = svext_f32(rw1, rw1, 4);
                    REDUCE_FULL_01_ROW(4, 0, rw0, rw1);
                    REDUCE_FULL_01_ROW(5, 1, rw0, rw1);
                    REDUCE_FULL_01_ROW(6, 2, rw0, rw1);
                    REDUCE_FULL_01_ROW(7, 3, rw0, rw1);
                    rw0 = svext_f32(rw0, rw0, 4);
                    rw1 = svext_f32(rw1, rw1, 4);
                    REDUCE_FULL_01_ROW(8, 0, rw0, rw1);
                    REDUCE_FULL_01_ROW(9, 1, rw0, rw1);
                    REDUCE_FULL_01_ROW(10, 2, rw0, rw1);
                    REDUCE_FULL_01_ROW(11, 3, rw0, rw1);
                    rw0 = svext_f32(rw0, rw0, 4);
                    rw1 = svext_f32(rw1, rw1, 4);
                    REDUCE_FULL_01_ROW(12, 0, rw0, rw1);
                    REDUCE_FULL_01_ROW(13, 1, rw0, rw1);
                    REDUCE_FULL_01_ROW(14, 2, rw0, rw1);
                    REDUCE_FULL_01_ROW(15, 3, rw0, rw1);
#undef REDUCE_FULL_01_ROW
                }

                // Collapse the first head wave before loading the second one.
                // This keeps the original reduction tree while reducing live
                // Z state across the following load-heavy dim loop.
                const svfloat32_t first_sum0 =
                    svadd_f32_m(pg_all_f32, acc00, acc10);
                const svfloat32_t first_sum1 =
                    svadd_f32_m(pg_all_f32, acc01, acc11);

                svfloat32_t acc20 = zero;
                svfloat32_t acc21 = zero;
                svfloat32_t acc30 = zero;
                svfloat32_t acc31 = zero;
                svzero_za();
                const uint32_t *__restrict phase_k_ptr2 = k_phase;
                const bfloat16_t *__restrict q_wave2 =
                    q_n + kQWaveStride;
#pragma unroll 4
                for (int64_t p = 0; p < kDimPairs; ++p) {
                    const bfloat16_t *__restrict q_base =
                        q_wave2 + p * kQWavePairStride;
                    const svuint32_t k0_u32 =
                        svld1_u32(pg_all_f32, phase_k_ptr2);
                    const svuint32_t k1_u32 =
                        svld1_u32(pg_all_f32,
                                  phase_k_ptr2 + kTileRows);
                    const svbfloat16_t k0 =
                        svreinterpret_bf16_u32(k0_u32);
                    const svbfloat16_t k1 =
                        svreinterpret_bf16_u32(k1_u32);
                    const svbfloat16_t q2 =
                        svld1_bf16(pg_heads_bf16, q_base);
                    const svbfloat16_t q3 = svld1_bf16(
                        pg_heads_bf16, q_base + 16 * 2);

                    svmopa_za32_bf16_m(
                        0, pg_heads_bf16, pg_heads_bf16, q2, k0);
                    svmopa_za32_bf16_m(
                        1, pg_heads_bf16, pg_heads_bf16, q2, k1);
                    svmopa_za32_bf16_m(
                        2, pg_heads_bf16, pg_heads_bf16, q3, k0);
                    svmopa_za32_bf16_m(
                        3, pg_heads_bf16, pg_heads_bf16, q3, k1);
                    phase_k_ptr2 += kPhaseTokens;
                }

                {
#define REDUCE_FULL_23_ROW(R, L, W2, W3)                                  \
                    do {                                                    \
                        svfloat32_t s20 = svread_hor_za32_f32_m(            \
                            zero, pg_all_f32, 0, (R));                      \
                        svfloat32_t s21 = svread_hor_za32_f32_m(            \
                            zero, pg_all_f32, 1, (R));                      \
                        svfloat32_t s30 = svread_hor_za32_f32_m(            \
                            zero, pg_all_f32, 2, (R));                      \
                        svfloat32_t s31 = svread_hor_za32_f32_m(            \
                            zero, pg_all_f32, 3, (R));                      \
                        s20 = svmax_n_f32_m(pg_all_f32, s20, 0.0f);         \
                        s21 = svmax_n_f32_m(pg_all_f32, s21, 0.0f);         \
                        s30 = svmax_n_f32_m(pg_all_f32, s30, 0.0f);         \
                        s31 = svmax_n_f32_m(pg_all_f32, s31, 0.0f);         \
                        const svfloat32_t w2 =                              \
                            svdup_lane_f32((W2), (L));                      \
                        const svfloat32_t w3 =                              \
                            svdup_lane_f32((W3), (L));                      \
                        acc20 = svmla_f32_m(                                \
                            pg_all_f32, acc20, s20, w2);                    \
                        acc21 = svmla_f32_m(                                \
                            pg_all_f32, acc21, s21, w2);                    \
                        acc30 = svmla_f32_m(                                \
                            pg_all_f32, acc30, s30, w3);                    \
                        acc31 = svmla_f32_m(                                \
                            pg_all_f32, acc31, s31, w3);                    \
                    } while (0)
                    svfloat32_t rw2 = weight2;
                    svfloat32_t rw3 = weight3;
                    REDUCE_FULL_23_ROW(0, 0, rw2, rw3);
                    REDUCE_FULL_23_ROW(1, 1, rw2, rw3);
                    REDUCE_FULL_23_ROW(2, 2, rw2, rw3);
                    REDUCE_FULL_23_ROW(3, 3, rw2, rw3);
                    rw2 = svext_f32(rw2, rw2, 4);
                    rw3 = svext_f32(rw3, rw3, 4);
                    REDUCE_FULL_23_ROW(4, 0, rw2, rw3);
                    REDUCE_FULL_23_ROW(5, 1, rw2, rw3);
                    REDUCE_FULL_23_ROW(6, 2, rw2, rw3);
                    REDUCE_FULL_23_ROW(7, 3, rw2, rw3);
                    rw2 = svext_f32(rw2, rw2, 4);
                    rw3 = svext_f32(rw3, rw3, 4);
                    REDUCE_FULL_23_ROW(8, 0, rw2, rw3);
                    REDUCE_FULL_23_ROW(9, 1, rw2, rw3);
                    REDUCE_FULL_23_ROW(10, 2, rw2, rw3);
                    REDUCE_FULL_23_ROW(11, 3, rw2, rw3);
                    rw2 = svext_f32(rw2, rw2, 4);
                    rw3 = svext_f32(rw3, rw3, 4);
                    REDUCE_FULL_23_ROW(12, 0, rw2, rw3);
                    REDUCE_FULL_23_ROW(13, 1, rw2, rw3);
                    REDUCE_FULL_23_ROW(14, 2, rw2, rw3);
                    REDUCE_FULL_23_ROW(15, 3, rw2, rw3);
#undef REDUCE_FULL_23_ROW
                }

                const svfloat32_t second_sum0 =
                    svadd_f32_m(pg_all_f32, acc20, acc30);
                const svfloat32_t result0 =
                    svadd_f32_m(pg_all_f32, first_sum0, second_sum0);
                const svfloat32_t second_sum1 =
                    svadd_f32_m(pg_all_f32, acc21, acc31);
                const svfloat32_t result1 =
                    svadd_f32_m(pg_all_f32, first_sum1, second_sum1);
                svst1_f32(pg_all_f32, output_n, result0);
                svst1_f32(pg_all_f32, output_n + kTileRows, result1);
            }
        }
    }
}

// A 16-token phase needs one ZA32 tile per 16-head group.  Using all four
// tiles for the four head groups lets each freshly packed K vector feed all
// 64 heads before it leaves a Z register.  Compared with the 32-token path,
// this halves the hot K scratch footprint and the number of K scratch loads;
// horizontal/vertical packing work and the total BFMOPA count stay unchanged.
template <bool WaveMajor, int64_t LocalNextN>
__arm_locally_streaming
__arm_new("za")
__attribute__((noinline, aligned(256),
               section(".text.indexer_mqa_phase16")))
void sme_bf16_phase16_four_head_tiles(
    const FullBlockTask *__restrict tasks,
    int64_t block_count,
    uint32_t *__restrict k_phase,
    const bfloat16_t *__restrict q_pair,
    const float *__restrict weights,
    float *__restrict output,
    int64_t output_stride,
    const int64_t *__restrict valid_lens)
{
    static_assert(LocalNextN == 1 || LocalNextN == 2);
    static_assert(WaveMajor == (LocalNextN == 1));
    constexpr uint64_t kTileRows = 16;
    constexpr int64_t kQPairStride =
        WaveMajor ? kHeads : kHeads * 2;
    constexpr int64_t kQWaveStride = kDimPairs * kHeads;
    const svbool_t pg_all_f32 = svwhilelt_b32_u64(0, kTileRows);
    const svbool_t pg_all_bf16 =
        svwhilelt_b16_u64(0, 2 * kTileRows);
    const svfloat32_t zero = svdup_n_f32(0.0f);
    // In the single-query specialization the same 64 weights are reused for
    // every block and phase. Keep the four 16-head vectors live for the whole
    // call so the hot epilogue does not reload them repeatedly.
    const svfloat32_t n1_weight0 = LocalNextN == 1
        ? svld1_f32(pg_all_f32, weights + 0 * kTileRows)
        : zero;
    const svfloat32_t n1_weight1 = LocalNextN == 1
        ? svld1_f32(pg_all_f32, weights + 1 * kTileRows)
        : zero;
    const svfloat32_t n1_weight2 = LocalNextN == 1
        ? svld1_f32(pg_all_f32, weights + 2 * kTileRows)
        : zero;
    const svfloat32_t n1_weight3 = LocalNextN == 1
        ? svld1_f32(pg_all_f32, weights + 3 * kTileRows)
        : zero;

    for (int64_t block_pos = 0; block_pos < block_count; ++block_pos) {
        const bfloat16_t *__restrict kv_block = tasks[block_pos].kv_block;
        const int64_t block_token_base = tasks[block_pos].token_base;

        for (int64_t token_phase = 0;
             token_phase < kBlockSize / kFourHeadPhaseTokens;
             ++token_phase) {
            const int64_t phase_token_base =
                token_phase * kFourHeadPhaseTokens;
            const bfloat16_t *__restrict src =
                kv_block + phase_token_base * kDim;

            for (uint32_t row = 0; row < kTileRows; ++row) {
                svld1_hor_za32(0, row, pg_all_f32, src + 0 * 32);
                svld1_hor_za32(1, row, pg_all_f32, src + 1 * 32);
                svld1_hor_za32(2, row, pg_all_f32, src + 2 * 32);
                svld1_hor_za32(3, row, pg_all_f32, src + 3 * 32);
                src += kDim;
            }

            #pragma clang loop unroll(disable)
            for (uint32_t column = 0; column < 16; column += 4) {
                svst1_ver_za32(
                    0, column + 0, pg_all_f32,
                    k_phase + (0 * 16 + column + 0) *
                        kFourHeadPhaseTokens);
                svst1_ver_za32(
                    0, column + 1, pg_all_f32,
                    k_phase + (0 * 16 + column + 1) *
                        kFourHeadPhaseTokens);
                svst1_ver_za32(
                    0, column + 2, pg_all_f32,
                    k_phase + (0 * 16 + column + 2) *
                        kFourHeadPhaseTokens);
                svst1_ver_za32(
                    0, column + 3, pg_all_f32,
                    k_phase + (0 * 16 + column + 3) *
                        kFourHeadPhaseTokens);
            }
            #pragma clang loop unroll(disable)
            for (uint32_t column = 0; column < 16; column += 4) {
                svst1_ver_za32(
                    1, column + 0, pg_all_f32,
                    k_phase + (1 * 16 + column + 0) *
                        kFourHeadPhaseTokens);
                svst1_ver_za32(
                    1, column + 1, pg_all_f32,
                    k_phase + (1 * 16 + column + 1) *
                        kFourHeadPhaseTokens);
                svst1_ver_za32(
                    1, column + 2, pg_all_f32,
                    k_phase + (1 * 16 + column + 2) *
                        kFourHeadPhaseTokens);
                svst1_ver_za32(
                    1, column + 3, pg_all_f32,
                    k_phase + (1 * 16 + column + 3) *
                        kFourHeadPhaseTokens);
            }
            #pragma clang loop unroll(disable)
            for (uint32_t column = 0; column < 16; column += 4) {
                svst1_ver_za32(
                    2, column + 0, pg_all_f32,
                    k_phase + (2 * 16 + column + 0) *
                        kFourHeadPhaseTokens);
                svst1_ver_za32(
                    2, column + 1, pg_all_f32,
                    k_phase + (2 * 16 + column + 1) *
                        kFourHeadPhaseTokens);
                svst1_ver_za32(
                    2, column + 2, pg_all_f32,
                    k_phase + (2 * 16 + column + 2) *
                        kFourHeadPhaseTokens);
                svst1_ver_za32(
                    2, column + 3, pg_all_f32,
                    k_phase + (2 * 16 + column + 3) *
                        kFourHeadPhaseTokens);
            }
            #pragma clang loop unroll(disable)
            for (uint32_t column = 0; column < 16; column += 4) {
                svst1_ver_za32(
                    3, column + 0, pg_all_f32,
                    k_phase + (3 * 16 + column + 0) *
                        kFourHeadPhaseTokens);
                svst1_ver_za32(
                    3, column + 1, pg_all_f32,
                    k_phase + (3 * 16 + column + 1) *
                        kFourHeadPhaseTokens);
                svst1_ver_za32(
                    3, column + 2, pg_all_f32,
                    k_phase + (3 * 16 + column + 2) *
                        kFourHeadPhaseTokens);
                svst1_ver_za32(
                    3, column + 3, pg_all_f32,
                    k_phase + (3 * 16 + column + 3) *
                        kFourHeadPhaseTokens);
            }

            if constexpr (WaveMajor) {
                if (block_pos + 1 < block_count) {
                    const bfloat16_t *__restrict next_block =
                        tasks[block_pos + 1].kv_block;
                    constexpr int64_t kBf16PerCacheLine = 32;
                    const int64_t begin = phase_token_base * kDim;
                    const int64_t end =
                        begin + kFourHeadPhaseTokens * kDim;
                    for (int64_t offset = begin; offset < end;
                         offset += kBf16PerCacheLine) {
                        __builtin_prefetch(next_block + offset, 0, 2);
                    }
                }
            }

            for (int64_t n = 0; n < LocalNextN; ++n) {
                uint64_t active_tokens = kFourHeadPhaseTokens;
                if constexpr (LocalNextN == 2) {
                    const int64_t remaining =
                        valid_lens[n] - block_token_base - phase_token_base;
                    if (remaining <= 0) {
                        continue;
                    }
                    active_tokens = static_cast<uint64_t>(
                        std::min<int64_t>(
                            kFourHeadPhaseTokens, remaining));
                }
                const svbool_t pg_tokens_f32 =
                    svwhilelt_b32_u64(0, active_tokens);
                const svbool_t pg_tokens_bf16 =
                    svwhilelt_b16_u64(0, 2 * active_tokens);
                const bfloat16_t *__restrict q_n =
                    q_pair + n * kDimPairs * kHeads * 2;
                const float *__restrict weights_n = weights + n * kHeads;
                float *__restrict output_n =
                    output + n * output_stride + block_token_base +
                    phase_token_base;

                svzero_za();
                const uint32_t *__restrict phase_k_ptr = k_phase;
                const bfloat16_t *__restrict q_wave2 =
                    q_n + (WaveMajor ? kQWaveStride : 2 * 16 * 2);
                #pragma unroll 4
                for (int64_t p = 0; p < kDimPairs; ++p) {
                    const bfloat16_t *__restrict q_base01 =
                        q_n + p * kQPairStride;
                    const bfloat16_t *__restrict q_base23 = WaveMajor
                        ? q_wave2 + p * kQPairStride
                        : q_base01 + 2 * 16 * 2;
                    const svuint32_t k_u32 =
                        svld1_u32(pg_tokens_f32, phase_k_ptr);
                    const svbfloat16_t k =
                        svreinterpret_bf16_u32(k_u32);
                    const svbfloat16_t q0 =
                        svld1_bf16(pg_all_bf16, q_base01);
                    const svbfloat16_t q1 =
                        svld1_bf16(pg_all_bf16, q_base01 + 16 * 2);
                    const svbfloat16_t q2 =
                        svld1_bf16(pg_all_bf16, q_base23);
                    const svbfloat16_t q3 =
                        svld1_bf16(pg_all_bf16, q_base23 + 16 * 2);

                    svmopa_za32_bf16_m(
                        0, pg_all_bf16, pg_tokens_bf16, q0, k);
                    svmopa_za32_bf16_m(
                        1, pg_all_bf16, pg_tokens_bf16, q1, k);
                    svmopa_za32_bf16_m(
                        2, pg_all_bf16, pg_tokens_bf16, q2, k);
                    svmopa_za32_bf16_m(
                        3, pg_all_bf16, pg_tokens_bf16, q3, k);
                    phase_k_ptr += kFourHeadPhaseTokens;
                }

                svfloat32_t acc0 = zero;
                svfloat32_t acc1 = zero;
                svfloat32_t acc2 = zero;
                svfloat32_t acc3 = zero;
                const svfloat32_t weight0 = LocalNextN == 1
                    ? n1_weight0
                    : svld1_f32(pg_all_f32, weights_n + 0 * kTileRows);
                const svfloat32_t weight1 = LocalNextN == 1
                    ? n1_weight1
                    : svld1_f32(pg_all_f32, weights_n + 1 * kTileRows);
                const svfloat32_t weight2 = LocalNextN == 1
                    ? n1_weight2
                    : svld1_f32(pg_all_f32, weights_n + 2 * kTileRows);
                const svfloat32_t weight3 = LocalNextN == 1
                    ? n1_weight3
                    : svld1_f32(pg_all_f32, weights_n + 3 * kTileRows);

#define REDUCE_PHASE16_ROW(R, L, W0, W1, W2, W3)                         \
                do {                                                       \
                    svfloat32_t s0 = svread_hor_za32_f32_m(                \
                        zero, pg_tokens_f32, 0, (R));                      \
                    svfloat32_t s1 = svread_hor_za32_f32_m(                \
                        zero, pg_tokens_f32, 1, (R));                      \
                    svfloat32_t s2 = svread_hor_za32_f32_m(                \
                        zero, pg_tokens_f32, 2, (R));                      \
                    svfloat32_t s3 = svread_hor_za32_f32_m(                \
                        zero, pg_tokens_f32, 3, (R));                      \
                    s0 = svmax_n_f32_m(pg_tokens_f32, s0, 0.0f);           \
                    s1 = svmax_n_f32_m(pg_tokens_f32, s1, 0.0f);           \
                    s2 = svmax_n_f32_m(pg_tokens_f32, s2, 0.0f);           \
                    s3 = svmax_n_f32_m(pg_tokens_f32, s3, 0.0f);           \
                    acc0 = svmla_lane_f32(acc0, s0, (W0), (L));           \
                    acc1 = svmla_lane_f32(acc1, s1, (W1), (L));           \
                    acc2 = svmla_lane_f32(acc2, s2, (W2), (L));           \
                    acc3 = svmla_lane_f32(acc3, s3, (W3), (L));           \
                } while (0)
                // Indexed FMLA selects a lane independently in each 128-bit
                // segment. Replicate one four-weight group into every segment
                // so all token lanes still use the same scalar head weight.
                svuint32_t weight_indices = svand_n_u32_x(
                    pg_all_f32, svindex_u32(0, 1), 3);
                svfloat32_t rw0 = svtbl_f32(weight0, weight_indices);
                svfloat32_t rw1 = svtbl_f32(weight1, weight_indices);
                svfloat32_t rw2 = svtbl_f32(weight2, weight_indices);
                svfloat32_t rw3 = svtbl_f32(weight3, weight_indices);
                REDUCE_PHASE16_ROW(0, 0, rw0, rw1, rw2, rw3);
                REDUCE_PHASE16_ROW(1, 1, rw0, rw1, rw2, rw3);
                REDUCE_PHASE16_ROW(2, 2, rw0, rw1, rw2, rw3);
                REDUCE_PHASE16_ROW(3, 3, rw0, rw1, rw2, rw3);
                weight_indices = svadd_n_u32_x(
                    pg_all_f32, weight_indices, 4);
                rw0 = svtbl_f32(weight0, weight_indices);
                rw1 = svtbl_f32(weight1, weight_indices);
                rw2 = svtbl_f32(weight2, weight_indices);
                rw3 = svtbl_f32(weight3, weight_indices);
                REDUCE_PHASE16_ROW(4, 0, rw0, rw1, rw2, rw3);
                REDUCE_PHASE16_ROW(5, 1, rw0, rw1, rw2, rw3);
                REDUCE_PHASE16_ROW(6, 2, rw0, rw1, rw2, rw3);
                REDUCE_PHASE16_ROW(7, 3, rw0, rw1, rw2, rw3);
                weight_indices = svadd_n_u32_x(
                    pg_all_f32, weight_indices, 4);
                rw0 = svtbl_f32(weight0, weight_indices);
                rw1 = svtbl_f32(weight1, weight_indices);
                rw2 = svtbl_f32(weight2, weight_indices);
                rw3 = svtbl_f32(weight3, weight_indices);
                REDUCE_PHASE16_ROW(8, 0, rw0, rw1, rw2, rw3);
                REDUCE_PHASE16_ROW(9, 1, rw0, rw1, rw2, rw3);
                REDUCE_PHASE16_ROW(10, 2, rw0, rw1, rw2, rw3);
                REDUCE_PHASE16_ROW(11, 3, rw0, rw1, rw2, rw3);
                weight_indices = svadd_n_u32_x(
                    pg_all_f32, weight_indices, 4);
                rw0 = svtbl_f32(weight0, weight_indices);
                rw1 = svtbl_f32(weight1, weight_indices);
                rw2 = svtbl_f32(weight2, weight_indices);
                rw3 = svtbl_f32(weight3, weight_indices);
                REDUCE_PHASE16_ROW(12, 0, rw0, rw1, rw2, rw3);
                REDUCE_PHASE16_ROW(13, 1, rw0, rw1, rw2, rw3);
                REDUCE_PHASE16_ROW(14, 2, rw0, rw1, rw2, rw3);
                REDUCE_PHASE16_ROW(15, 3, rw0, rw1, rw2, rw3);
#undef REDUCE_PHASE16_ROW

                const svfloat32_t first_sum =
                    svadd_f32_m(pg_tokens_f32, acc0, acc1);
                const svfloat32_t second_sum =
                    svadd_f32_m(pg_tokens_f32, acc2, acc3);
                const svfloat32_t result =
                    svadd_f32_m(pg_tokens_f32, first_sum, second_sum);
                svst1_f32(pg_tokens_f32, output_n, result);
            }
        }
    }
}

// The active 512-bit full pipeline only needs one 16-token K phase.  Keep
// the 64-token gather layout in a separate cold frame so it does not enlarge
// every worker's hot partition stack.
__attribute__((noinline))
void indexer_mqa_run_fallback_block(
    const bfloat16_t *__restrict kv_block,
    int64_t context_tokens,
    int64_t token_base,
    const bfloat16_t *__restrict q_pair,
    const float *__restrict w_tile,
    float *__restrict batch_output,
    int64_t output_stride,
    const int64_t *__restrict valid_lens,
    int64_t next_n,
    uint64_t gather_vlw)
{
    alignas(64) uint32_t k_pairs[kDimPairs][kBlockSize];
    const svbool_t pg_all_f32 = svptrue_b32();
    const svint32_t gather_offsets = svlsl_n_s32_x(
        pg_all_f32, svindex_s32(0, 1), 8);

    for (int64_t p = 0; p < kDimPairs; ++p) {
        for (uint64_t t = 0;
             t < static_cast<uint64_t>(context_tokens);
             t += gather_vlw) {
            const svbool_t pg = svwhilelt_b32(
                t, static_cast<uint64_t>(context_tokens));
            const uint32_t *__restrict kv_pair_base =
                reinterpret_cast<const uint32_t *>(
                    kv_block + t * kDim + p * 2);
            const svuint32_t pairs = svld1_gather_s32offset_u32(
                pg, kv_pair_base, gather_offsets);
            svst1_u32(pg, &k_pairs[p][t], pairs);
        }
    }

    for (int64_t n = 0; n < next_n; ++n) {
        if (token_base >= valid_lens[n]) {
            continue;
        }
        const int64_t tokens_this_n = std::min<int64_t>(
            context_tokens, valid_lens[n] - token_base);
        float *__restrict out_row =
            batch_output + n * output_stride + token_base;
        sme_bf16_block_reduce(
            q_pair + n * kDimPairs * kHeads * 2,
            &k_pairs[0][0], w_tile + n * kHeads,
            out_row, tokens_this_n);
    }
}

} // namespace indexer_mqa_detail

template <bool UseN1Pipeline>
inline void indexer_mqa_run_partition(
    const indexer_mqa_detail::IndexerTaskArgs &task,
    int64_t global_begin,
    int64_t global_end)
{
    using namespace indexer_mqa_detail;

    const bfloat16_t *__restrict q_data = task.q_data;
    const bfloat16_t *__restrict kv_data = task.kv_data;
    const int64_t *__restrict block_table_data = task.block_table_data;
    const int64_t *__restrict context_len_data = task.context_len_data;
    const float *__restrict weight_data = task.weight_data;
    float *__restrict out_data = task.out_data;
    const int64_t *__restrict block_prefix = task.block_prefix;
    const int64_t batch_size = task.batch_size;
    constexpr int64_t next_n = UseN1Pipeline ? 1 : 2;
    FLASH_ASSERT(task.next_n == next_n);
    const int64_t max_num_blocks = task.max_num_blocks;
    const int64_t max_model_len = task.max_model_len;
    const int64_t total_blocks = task.total_blocks;

    FLASH_ASSERT(global_begin >= 0 && global_begin <= global_end);
    FLASH_ASSERT(global_end <= total_blocks);
    // The n1 specialization cannot address a second query.  Removing that
    // unused 16 KiB region keeps the private hot workspace materially smaller.
    constexpr int64_t kLocalNextN = UseN1Pipeline ? 1 : kMaxNextN;
    QKScratchColor64<kLocalNextN> qk_scratch;
    auto &q_pair = qk_scratch.q_pair;
    auto &k_phase = qk_scratch.k_phase;
    alignas(64) float w_tile[kLocalNextN][kHeads];
    int64_t valid_lens[kMaxNextN];
    alignas(64) FullBlockTask full_tasks[kFullTaskPanel];

    int64_t global_block = global_begin;                                 
    int64_t batch_idx = static_cast<int64_t>(
        std::upper_bound(block_prefix, block_prefix + batch_size + 1,
                         global_block) - block_prefix - 1);

    while (global_block < global_end) {
            while (batch_idx < batch_size &&
                   global_block >= block_prefix[batch_idx + 1]) {
                ++batch_idx;
            }
            FLASH_ASSERT(batch_idx < batch_size);

            const int64_t context_len = context_len_data[batch_idx];
            const int64_t num_blocks =
                block_prefix[batch_idx + 1] - block_prefix[batch_idx];
            const int64_t block_begin =
                global_block - block_prefix[batch_idx];
            const int64_t block_end = std::min<int64_t>(
                num_blocks, global_end - block_prefix[batch_idx]);
            const bool owns_batch_tail = block_end == num_blocks;

            const svbool_t q_pg_all_f32 = svptrue_b32();
            const svbool_t q_pg_all_u64 = svptrue_b64();
            const svint64_t q_gather_offsets_u64 = svlsl_n_s64_x(
                q_pg_all_u64, svindex_s64(0, 1), 8);
            const uint64_t gather_vlw = svcntw();
            // Case 2's two head waves are consumed separately.  At the
            // target VL, store each wave in its own [pair][32-head] slab so
            // each SME wave walks 128 B per pair instead of skipping 256 B.
            const bool use_n1_wave_q = UseN1Pipeline && gather_vlw == 16 &&
                max_model_len % kBlockSize == 0;

            for (int64_t n = 0; n < next_n; ++n) {
                valid_lens[n] = context_len - next_n + n + 1;

                const int64_t q_base =
                    ((batch_idx * next_n + n) * kHeads) * kDim;
                const int64_t w_base = (batch_idx * next_n + n) * kHeads;

                if constexpr (UseN1Pipeline) {
                    const bfloat16_t *__restrict q_src = q_data + q_base;
                    bfloat16_t *__restrict q_dst = &q_pair[n][0][0];
                    if (use_n1_wave_q) {
                        pack_q_pairs<true>(
                            q_src, q_dst, q_pg_all_f32, q_pg_all_u64,
                            q_gather_offsets_u64);
                    } else {
                        pack_q_pairs<false>(
                            q_src, q_dst, q_pg_all_f32, q_pg_all_u64,
                            q_gather_offsets_u64);
                    }
                } else {
                    for (int64_t p = 0; p < kDimPairs; p += 2) {
                        for (int64_t h = 0; h < kHeads; h += 16) {
                            const uint64_t *__restrict q_heads_lo =
                                reinterpret_cast<const uint64_t *>(
                                    q_data + q_base + h * kDim + p * 2);
                            const uint64_t *__restrict q_heads_hi =
                                reinterpret_cast<const uint64_t *>(
                                    q_data + q_base +
                                    (h + 8) * kDim + p * 2);
                            const svuint32_t packed_lo =
                                svreinterpret_u32_u64(
                                    svld1_gather_s64offset_u64(
                                        q_pg_all_u64, q_heads_lo,
                                        q_gather_offsets_u64));
                            const svuint32_t packed_hi =
                                svreinterpret_u32_u64(
                                    svld1_gather_s64offset_u64(
                                        q_pg_all_u64, q_heads_hi,
                                        q_gather_offsets_u64));
                            svst1_u32(q_pg_all_f32,
                                reinterpret_cast<uint32_t *>(
                                    &q_pair[n][p][h * 2]),
                                svuzp1_u32(packed_lo, packed_hi));
                            svst1_u32(q_pg_all_f32,
                                reinterpret_cast<uint32_t *>(
                                    &q_pair[n][p + 1][h * 2]),
                                svuzp2_u32(packed_lo, packed_hi));
                        }
                    }
                }

                if constexpr (UseN1Pipeline) {
                    for (int64_t h = 0; h < kHeads; h += 16) {
                        const svfloat32_t weight_vector = svld1_f32(
                            q_pg_all_f32, weight_data + w_base + h);
                        svst1_f32(
                            q_pg_all_f32, &w_tile[n][h], weight_vector);
                    }
                } else {
                    for (int64_t h = 0; h < kHeads; ++h) {
                        w_tile[n][h] = weight_data[w_base + h];
                    }
                }

            }

            int64_t full_block_count = 0;
            float *__restrict batch_output =
                out_data + batch_idx * next_n * max_model_len;
            for (int64_t block_pos = block_begin;
                 block_pos < block_end; ++block_pos) {
                // test14's measured sort-off logical order is preserved.
                const int64_t b = block_pos;
                const int64_t token_base = b * kBlockSize;
                const int64_t context_tokens =
                    std::min<int64_t>(kBlockSize, context_len - token_base);
                const int64_t block_idx =
                    block_table_data[batch_idx * max_num_blocks + b];

                if (block_idx < 0) {
                    for (int64_t n = 0; n < next_n; ++n) {
                        if (token_base >= valid_lens[n]) {
                            continue;
                        }
                        const int64_t tokens_this_n = std::min<int64_t>(
                            context_tokens, valid_lens[n] - token_base);
                        float *__restrict out_row =
                            out_data + (batch_idx * next_n + n) * max_model_len + token_base;
                        std::fill(out_row, out_row + tokens_this_n, 0.0f);
                    }
                    continue;
                }

                const bfloat16_t *__restrict kv_block =
                    kv_data + block_idx * kBlockSize * kDim;

                // Case 1 keeps test10's full-pipeline tail route.  At the
                // official padded output stride, Case 2 now routes its final
                // partial physical block through the same wave-major n1 path.
                const bool is_pipeline_block = gather_vlw == 16 &&
                    (!UseN1Pipeline ||
                     (use_n1_wave_q &&
                      token_base + kBlockSize <= max_model_len));
                if (__builtin_expect(is_pipeline_block, 1)) {
                    full_tasks[full_block_count] = {kv_block, token_base};
                    ++full_block_count;
                    if constexpr (UseN1Pipeline) {
                        if (__builtin_expect(
                                full_block_count == kFullTaskPanel, 0)) {
                            sme_bf16_phase16_four_head_tiles<true, 1>(
                                full_tasks, full_block_count, &k_phase[0][0],
                                &q_pair[0][0][0], &w_tile[0][0],
                                batch_output, max_model_len, nullptr);
                            full_block_count = 0;
                        }
                    }
                    continue;
                }

                indexer_mqa_run_fallback_block(
                    kv_block, context_tokens, token_base,
                    &q_pair[0][0][0], &w_tile[0][0], batch_output,
                    max_model_len, valid_lens, next_n, gather_vlw);
            }

            if (full_block_count > 0) {
                if constexpr (UseN1Pipeline) {
                    sme_bf16_phase16_four_head_tiles<true, 1>(
                        full_tasks, full_block_count, &k_phase[0][0],
                        &q_pair[0][0][0], &w_tile[0][0],
                        batch_output, max_model_len, nullptr);
                } else {
                    sme_bf16_full_range_pipeline(
                        full_tasks, full_block_count, &k_phase[0][0],
                        &q_pair[0][0][0], &w_tile[0][0],
                        batch_output, max_model_len, valid_lens, next_n);
                }
            }

            // Exactly one partition owns this tail.  Fill it after the hot
            // KV/BFMOPA path so these streaming output stores cannot evict the
            // Q and weight tiles that were just packed into the local cache.
            if (owns_batch_tail) {
                for (int64_t n = 0; n < next_n; ++n) {
                    float *__restrict out_row =
                        out_data + (batch_idx * next_n + n) * max_model_len;
                    std::fill(out_row + valid_lens[n],
                              out_row + max_model_len, -INFINITY);
                }
            }

            global_block = block_prefix[batch_idx] + block_end;
            ++batch_idx;
        }
}

#if defined(INDEXER_KUPL)
static void indexer_mqa_kupl_partition(
    kupl_nd_range_t *range,
    void *args,
    int tid,
    int tnum)
{
    FLASH_ASSERT(tid >= 0 && tid < tnum);
    FLASH_ASSERT(tnum == indexer_mqa_detail::kFixedThreads);
    const auto &r = range->nd_range[0];
    const auto &task = *static_cast<
        const indexer_mqa_detail::IndexerTaskArgs *>(args);
    if (task.next_n == 1) {
        for (int64_t slot = r.lower; slot < r.upper; slot += r.step) {
            const int64_t global_begin =
                task.total_blocks * slot /
                indexer_mqa_detail::kFixedThreads;
            const int64_t global_end =
                task.total_blocks * (slot + 1) /
                indexer_mqa_detail::kFixedThreads;
            indexer_mqa_run_partition<true>(
                task, global_begin, global_end);
        }
    } else {
        FLASH_ASSERT(task.next_n == 2);
        for (int64_t slot = r.lower; slot < r.upper; slot += r.step) {
            const int64_t global_begin =
                task.total_blocks * slot /
                indexer_mqa_detail::kFixedThreads;
            const int64_t global_end =
                task.total_blocks * (slot + 1) /
                indexer_mqa_detail::kFixedThreads;
            indexer_mqa_run_partition<false>(
                task, global_begin, global_end);
        }
    }
}
#endif

inline void indexer_bf16_paged_mqa_logits(
    const Tensor<bfloat16_t, 4>& q,
    const Tensor<bfloat16_t, 4>& kv_cache,
    const Tensor<int64_t, 2>& block_tables,
    const Tensor<int64_t, 1>& context_lens,
    const Tensor<float, 2>& weights,
    const Tensor<float, 2>& output,
    int64_t batch_size,
    int64_t next_n,
    int64_t num_heads,
    int64_t dim,
    int64_t block_size,
    int64_t max_model_len)
{
    using namespace indexer_mqa_detail;

    (void)num_heads;
    (void)dim;
    (void)block_size;

    const bfloat16_t *__restrict q_data = q.data_ptr();
    const bfloat16_t *__restrict kv_data = kv_cache.data_ptr();
    const int64_t *__restrict block_table_data = block_tables.data_ptr();
    const int64_t *__restrict context_len_data = context_lens.data_ptr();
    const float *__restrict weight_data = weights.data_ptr();
    float *__restrict out_data = output.data_ptr();
    const int64_t max_num_blocks = block_tables.sizes()[1];

    FLASH_ASSERT(batch_size > 0 && batch_size <= kMaxBatchSize);

    alignas(64) int64_t block_prefix[kMaxBatchSize + 1];
    block_prefix[0] = 0;
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        const int64_t context_len = context_len_data[batch];
        const int64_t num_blocks =
            (context_len + kBlockSize - 1) / kBlockSize;
        block_prefix[batch + 1] = block_prefix[batch] + num_blocks;
    }

    const int64_t total_blocks = block_prefix[batch_size];
    FLASH_ASSERT(total_blocks > 0);

    IndexerTaskArgs task = {
        q_data, kv_data, block_table_data, context_len_data,
        weight_data, out_data, block_prefix, batch_size, next_n,
        max_num_blocks, max_model_len, total_blocks};

#if defined(INDEXER_KUPL)
    // With the pthread backend this count is derived from the caller's CPU
    // affinity during KUPL initialization.  The runner preserves taskset's
    // full 0-31 mask until this point.
    FLASH_ASSERT(kupl_get_num_executors() >= kFixedThreads);
    kupl_nd_range_t range;
    KUPL_1D_RANGE_INIT(range, 0, kFixedThreads);
    kupl_parallel_for_desc_t desc{};
    desc.field_mask = KUPL_PARALLEL_FOR_DESC_FIELD_DEFAULT;
    desc.range = &range;
    desc.egroup = nullptr;
    desc.concurrency = kFixedThreads;
    desc.policy = KUPL_LOOP_POLICY_STATIC;
    const int status = kupl_parallel_for(
        &desc, indexer_mqa_kupl_partition, &task);
    FLASH_ASSERT(status == KUPL_OK);
#else
    FLASH_ASSERT(omp_get_max_threads() >= kFixedThreads);
    #pragma omp parallel num_threads(kFixedThreads)
    {
        const int thread_id = omp_get_thread_num();
        const int64_t global_begin =
            total_blocks * thread_id / kFixedThreads;
        const int64_t global_end =
            total_blocks * (thread_id + 1) / kFixedThreads;
        if (next_n == 1) {
            indexer_mqa_run_partition<true>(
                task, global_begin, global_end);
        } else {
            FLASH_ASSERT(next_n == 2);
            indexer_mqa_run_partition<false>(
                task, global_begin, global_end);
        }
    }
#endif
}
