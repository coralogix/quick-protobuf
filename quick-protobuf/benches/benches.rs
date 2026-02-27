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
