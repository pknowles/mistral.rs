#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Result, Tensor};
use mistralrs_paged_attn::{kv_scale_update, paged_attention};

const NUM_HEADS: usize = 40;
const KV_HEADS: usize = 8;
const HEAD_SIZE: usize = 128;
const BLOCK_SIZE: usize = 32;
const MAX_CONTEXT: usize = 2048;
const CACHE_VECTOR_BYTES: usize = 16;
const CASES: [(usize, usize); 6] = [
    (16, 2048),
    (16, 1152),
    (15, 2048),
    (14, 512),
    (16, 2048),
    (32, 1024),
];
const TOLERANCE: f32 = 0.01;

#[test]
fn kv_scale_update_covers_the_last_partial_block() -> Result<()> {
    const ELEMENTS: usize = 1025;
    const SCALE_TOLERANCE: f32 = 4.0 * f32::EPSILON;
    let device = Device::new_cuda(0)?;
    for dtype in [DType::BF16, DType::F16, DType::F32] {
        let mut keys = vec![0.0f32; ELEMENTS];
        let mut values = vec![0.0f32; ELEMENTS];
        keys[ELEMENTS - 1] = -480.0;
        values[ELEMENTS - 1] = 720.0;
        let keys = Tensor::from_vec(keys, ELEMENTS, &device)?.to_dtype(dtype)?;
        let values = Tensor::from_vec(values, ELEMENTS, &device)?.to_dtype(dtype)?;
        let k_scale = Tensor::zeros(1, DType::F32, &device)?;
        let v_scale = Tensor::zeros(1, DType::F32, &device)?;
        kv_scale_update(&keys, &values, &k_scale, &v_scale)?;
        assert!(
            (k_scale.to_vec1::<f32>()?[0] - 2.0).abs() <= SCALE_TOLERANCE,
            "{dtype:?}"
        );
        assert!(
            (v_scale.to_vec1::<f32>()?[0] - 3.0).abs() <= SCALE_TOLERANCE,
            "{dtype:?}"
        );
    }
    Ok(())
}

#[test]
fn paged_attention_handles_context_length_reordering() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let num_blocks = MAX_CONTEXT / BLOCK_SIZE;
    for dtype in [DType::BF16, DType::F16, DType::F32] {
        let vector_width = CACHE_VECTOR_BYTES / dtype.size_in_bytes();
        let keys = Tensor::zeros(
            (
                num_blocks,
                KV_HEADS,
                HEAD_SIZE / vector_width,
                BLOCK_SIZE,
                vector_width,
            ),
            dtype,
            &device,
        )?;
        let values = Tensor::ones(
            (num_blocks, KV_HEADS, HEAD_SIZE, BLOCK_SIZE),
            dtype,
            &device,
        )?;
        for (batch, context_len) in CASES {
            let query = Tensor::zeros((batch, NUM_HEADS, HEAD_SIZE), dtype, &device)?;
            let blocks = Tensor::from_vec(
                (0..num_blocks as u32)
                    .cycle()
                    .take(batch * num_blocks)
                    .collect(),
                (batch, num_blocks),
                &device,
            )?;
            let lengths = Tensor::from_vec(vec![context_len as u32; batch], batch, &device)?;
            let output = paged_attention(
                &query,
                None,
                None,
                &keys,
                &values,
                &blocks,
                &lengths,
                None,
                context_len,
                1.0 / (HEAD_SIZE as f32).sqrt(),
                1.0,
                None,
            )?;
            let output = output
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert!(
                output.iter().all(|value| (value - 1.0).abs() < TOLERANCE),
                "{dtype:?}, batch {batch}, context {context_len}"
            );
        }
    }
    Ok(())
}
