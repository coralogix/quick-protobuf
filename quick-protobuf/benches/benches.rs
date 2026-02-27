#![feature(test)]

extern crate quick_protobuf;
extern crate test;

#[macro_use]
extern crate lazy_static;

use test::{black_box, Bencher};

use quick_protobuf::{BytesReader, Writer};

const LEN: i32 = 10_000;
const PACKED_FIELDS: usize = 100;
const PACKED_ELEMS: i32 = 100;

lazy_static! {
    static ref BUFFER: Vec<u8> = {
        let mut buf = Vec::new();
        {
            let mut writer = Writer::new(&mut buf);
            for i in 0..LEN {
                writer.write_int32(i).unwrap();
            }
        }
        buf
    };

    static ref FIXED32_BUFFER: Vec<u8> = {
        let mut buf = Vec::new();
        {
            let mut writer = Writer::new(&mut buf);
            for i in 0..LEN as u32 {
                writer.write_fixed32(i).unwrap();
            }
        }
        buf
    };

    static ref FIXED64_BUFFER: Vec<u8> = {
        let mut buf = Vec::new();
        {
            let mut writer = Writer::new(&mut buf);
            for i in 0..LEN as u64 {
                writer.write_fixed64(i).unwrap();
            }
        }
        buf
    };

    static ref STRING_BUFFER: Vec<u8> = {
        let mut buf = Vec::new();
        {
            let mut writer = Writer::new(&mut buf);
            for _ in 0..LEN {
                writer.write_string("hello world").unwrap();
            }
        }
        buf
    };

    static ref PACKED_BUFFER: Vec<u8> = {
        let mut buf = Vec::new();
        {
            let mut writer = Writer::new(&mut buf);
            let values: Vec<i32> = (0..PACKED_ELEMS).collect();
            for _ in 0..PACKED_FIELDS {
                writer
                    .write_packed(
                        &values,
                        |w, &v| w.write_int32(v),
                        &|&v| quick_protobuf::sizeofs::sizeof_varint(v as u64),
                    )
                    .unwrap();
            }
        }
        buf
    };

    // Packed fields with mixed varint sizes (1-4 byte varints)
    static ref PACKED_MIXED_BUFFER: Vec<u8> = {
        let mut buf = Vec::new();
        {
            let mut writer = Writer::new(&mut buf);
            let values: Vec<i32> = (0..PACKED_ELEMS)
                .map(|i| match i % 4 {
                    0 => (i % 127) as i32,           // 1-byte varint
                    1 => 200 + i as i32,              // 2-byte varint
                    2 => 20_000 + (i * 100) as i32,   // 3-byte varint
                    _ => 3_000_000 + (i * 1000) as i32, // 4-byte varint
                })
                .collect();
            for _ in 0..PACKED_FIELDS {
                writer
                    .write_packed(
                        &values,
                        |w, &v| w.write_int32(v),
                        &|&v| quick_protobuf::sizeofs::sizeof_varint(v as u64),
                    )
                    .unwrap();
            }
        }
        buf
    };

    // Packed fields with small values only (1-byte varints, best case for batch decode)
    static ref PACKED_SMALL_BUFFER: Vec<u8> = {
        let mut buf = Vec::new();
        {
            let mut writer = Writer::new(&mut buf);
            let values: Vec<i32> = (0..PACKED_ELEMS).map(|i| (i % 127) as i32).collect();
            for _ in 0..PACKED_FIELDS {
                writer
                    .write_packed(
                        &values,
                        |w, &v| w.write_int32(v),
                        &|&v| quick_protobuf::sizeofs::sizeof_varint(v as u64),
                    )
                    .unwrap();
            }
        }
        buf
    };

    // Packed fields with large values (4-5 byte varints)
    static ref PACKED_LARGE_BUFFER: Vec<u8> = {
        let mut buf = Vec::new();
        {
            let mut writer = Writer::new(&mut buf);
            let values: Vec<i32> = (0..PACKED_ELEMS)
                .map(|i| if i % 2 == 0 {
                    3_000_000 + (i * 10_000) as i32    // 4-byte varint
                } else {
                    300_000_000 + (i * 1_000_000) as i32 // 5-byte varint
                })
                .collect();
            for _ in 0..PACKED_FIELDS {
                writer
                    .write_packed(
                        &values,
                        |w, &v| w.write_int32(v),
                        &|&v| quick_protobuf::sizeofs::sizeof_varint(v as u64),
                    )
                    .unwrap();
            }
        }
        buf
    };

    // Non-packed sequence of multi-byte varints (2-4 bytes each)
    static ref MULTIBYTE_VARINT32_BUFFER: Vec<u8> = {
        let mut buf = Vec::new();
        {
            let mut writer = Writer::new(&mut buf);
            for i in 0..LEN {
                let val = match i % 3 {
                    0 => 200 + i,                  // 2-byte varint
                    1 => 20_000 + i * 100,         // 3-byte varint
                    _ => 3_000_000 + i * 1000,     // 4-byte varint
                };
                writer.write_int32(val).unwrap();
            }
        }
        buf
    };
}

#[bench]
fn read_varint32(b: &mut Bencher) {
    b.iter(|| {
        let mut reader = BytesReader::from_bytes(&BUFFER);
        for _ in 0..LEN {
            let _ = black_box(reader.read_varint32(&BUFFER).unwrap());
        }
        assert!(reader.is_eof());
    })
}

#[bench]
fn read_varint64(b: &mut Bencher) {
    b.iter(|| {
        let mut reader = BytesReader::from_bytes(&BUFFER);
        for _ in 0..LEN {
            let _ = black_box(reader.read_varint64(&BUFFER).unwrap());
        }
        assert!(reader.is_eof());
    })
}

#[bench]
fn read_varint64_and_is_eof(b: &mut Bencher) {
    b.iter(|| {
        let mut reader = BytesReader::from_bytes(&BUFFER);
        for _ in 0..LEN {
            assert!(!reader.is_eof());
            let _ = black_box(reader.read_varint64(&BUFFER).unwrap());
        }
    })
}

#[bench]
fn read_fixed32(b: &mut Bencher) {
    b.iter(|| {
        let mut reader = BytesReader::from_bytes(&FIXED32_BUFFER);
        for _ in 0..LEN {
            let _ = black_box(reader.read_fixed32(&FIXED32_BUFFER).unwrap());
        }
        assert!(reader.is_eof());
    })
}

#[bench]
fn read_fixed64(b: &mut Bencher) {
    b.iter(|| {
        let mut reader = BytesReader::from_bytes(&FIXED64_BUFFER);
        for _ in 0..LEN {
            let _ = black_box(reader.read_fixed64(&FIXED64_BUFFER).unwrap());
        }
        assert!(reader.is_eof());
    })
}

#[bench]
fn read_string(b: &mut Bencher) {
    b.iter(|| {
        let mut reader = BytesReader::from_bytes(&STRING_BUFFER);
        for _ in 0..LEN {
            let _ = black_box(reader.read_string(&STRING_BUFFER).unwrap());
        }
        assert!(reader.is_eof());
    })
}

#[bench]
fn read_packed_varint(b: &mut Bencher) {
    b.iter(|| {
        let mut reader = BytesReader::from_bytes(&PACKED_BUFFER);
        for _ in 0..PACKED_FIELDS {
            let v: Vec<i32> = black_box(
                reader
                    .read_packed(&PACKED_BUFFER, |r, b| r.read_int32(b))
                    .unwrap(),
            );
            assert_eq!(v.len(), PACKED_ELEMS as usize);
        }
        assert!(reader.is_eof());
    })
}

#[bench]
fn read_packed_int32(b: &mut Bencher) {
    b.iter(|| {
        let mut reader = BytesReader::from_bytes(&PACKED_MIXED_BUFFER);
        for _ in 0..PACKED_FIELDS {
            let v: Vec<i32> = black_box(
                reader.read_packed_int32(&PACKED_MIXED_BUFFER).unwrap(),
            );
            assert_eq!(v.len(), PACKED_ELEMS as usize);
        }
        assert!(reader.is_eof());
    })
}

#[bench]
fn read_packed_int32_small_values(b: &mut Bencher) {
    b.iter(|| {
        let mut reader = BytesReader::from_bytes(&PACKED_SMALL_BUFFER);
        for _ in 0..PACKED_FIELDS {
            let v: Vec<i32> = black_box(
                reader.read_packed_int32(&PACKED_SMALL_BUFFER).unwrap(),
            );
            assert_eq!(v.len(), PACKED_ELEMS as usize);
        }
        assert!(reader.is_eof());
    })
}

#[bench]
fn read_packed_int32_large_values(b: &mut Bencher) {
    b.iter(|| {
        let mut reader = BytesReader::from_bytes(&PACKED_LARGE_BUFFER);
        for _ in 0..PACKED_FIELDS {
            let v: Vec<i32> = black_box(
                reader.read_packed_int32(&PACKED_LARGE_BUFFER).unwrap(),
            );
            assert_eq!(v.len(), PACKED_ELEMS as usize);
        }
        assert!(reader.is_eof());
    })
}

#[bench]
fn read_varint32_multibyte(b: &mut Bencher) {
    b.iter(|| {
        let mut reader = BytesReader::from_bytes(&MULTIBYTE_VARINT32_BUFFER);
        for _ in 0..LEN {
            let _ = black_box(reader.read_varint32(&MULTIBYTE_VARINT32_BUFFER).unwrap());
        }
        assert!(reader.is_eof());
    })
}
