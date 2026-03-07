use super::*;

fn bench_data(bytes: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes);
    let mut state: u32 = 0x1234_5678;
    while out.len() < bytes {
        let zone = (out.len() / 4096) % 3;
        match zone {
            0 => out.extend_from_slice(b"cozip-cpu-gpu-deflate-"),
            1 => out.extend_from_slice(b"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
            _ => {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                out.push((state >> 24) as u8);
            }
        }
    }
    out.truncate(bytes);
    out
}

#[test]
fn pdeflate_roundtrip_cpu_only() {
    let input = bench_data(2 * 1024 * 1024 + 123);
    let options = HybridOptions {
        gpu_compress_enabled: false,
        gpu_decompress_enabled: false,
        gpu_decompress_force_gpu: false,
        ..HybridOptions::default()
    };
    let cozip = CoZipDeflate::init(options).expect("init should succeed");

    let mut src = std::io::Cursor::new(input.clone());
    let mut compressed = Vec::new();
    let compress = cozip
        .deflate_compress_stream_zip_compatible_with_index(&mut src, &mut compressed)
        .expect("compression should succeed");

    let mut restored = Vec::new();
    let stats = cozip
        .pdeflate_decompress_bytes(&compressed, &mut restored)
        .expect("decompression should succeed");

    assert_eq!(restored, input);
    assert_eq!(compress.stats.chunk_count, stats.chunk_count);
    assert_eq!(compress.index, None);
}

#[test]
fn pdeflate_roundtrip_gpu_preferred() {
    let input = bench_data(1024 * 1024 + 321);
    let options = HybridOptions {
        gpu_compress_enabled: true,
        gpu_decompress_enabled: true,
        ..HybridOptions::default()
    };
    let cozip = CoZipDeflate::init(options).expect("init should succeed");

    let mut src = std::io::Cursor::new(input.clone());
    let mut compressed = Vec::new();
    cozip
        .deflate_compress_stream_zip_compatible(&mut src, &mut compressed)
        .expect("compression should succeed");

    let mut restored = Vec::new();
    cozip
        .pdeflate_decompress_bytes(&compressed, &mut restored)
        .expect("decompression should succeed");

    assert_eq!(restored, input);
}

#[test]
fn indexed_decompress_alias_uses_pdeflate_stream() {
    let input = bench_data(512 * 1024 + 17);
    let cozip = CoZipDeflate::init(HybridOptions::default()).expect("init should succeed");

    let mut src = std::io::Cursor::new(input.clone());
    let mut compressed = Vec::new();
    cozip
        .deflate_compress_stream_zip_compatible(&mut src, &mut compressed)
        .expect("compression should succeed");

    let mut decoded = Vec::new();
    let mut reader = std::io::Cursor::new(compressed);
    let stats = cozip
        .deflate_decompress_stream_zip_compatible_with_index(
            &mut reader,
            &mut decoded,
            &DeflateChunkIndex,
        )
        .expect("indexed alias should succeed");

    assert_eq!(decoded, input);
    assert_eq!(stats.output_bytes as usize, input.len());
}
