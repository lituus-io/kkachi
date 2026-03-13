// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Zero-copy buffer system for efficient memory management
//!
//! Provides [`Buffer`] for managing data with zero-copy semantics through:
//! - Memory-mapped files (true zero-copy from disk)
//! - Reference-counted bytes (zero-copy sharing)
//! - Static data (compile-time constants)

use bytes::Bytes;
use memmap2::Mmap;
use std::fs::File;
use std::io;
use std::ops::Range;
use std::path::Path;
use std::str::Utf8Error;
use std::sync::Arc;

/// Zero-copy buffer backed by memory-mapped file or allocated memory.
///
/// All variants provide O(1) access to the underlying data without copying.
/// The buffer can be sliced to create views without allocation.
#[derive(Clone)]
pub enum Buffer {
    /// Memory-mapped file - true zero-copy from disk
    Mmap(Arc<Mmap>),
    /// Reference-counted bytes - zero-copy sharing between threads
    Bytes(Bytes),
    /// Static data - compile-time constants
    Static(&'static [u8]),
    /// Empty buffer
    Empty,
}

impl Buffer {
    /// Create a buffer from a memory-mapped file.
    ///
    /// # Safety
    /// The file must not be modified while the mapping is active.
    ///
    /// # Errors
    /// Returns an error if the file cannot be opened or mapped.
    pub fn mmap(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = File::open(path)?;
        // SAFETY: We assume the file won't be modified while mapped
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Buffer::Mmap(Arc::new(mmap)))
    }

    /// Create a buffer from owned bytes.
    #[inline]
    pub fn from_bytes(bytes: impl Into<Bytes>) -> Self {
        Buffer::Bytes(bytes.into())
    }

    /// Create a buffer from a static slice.
    #[inline]
    pub const fn from_static(data: &'static [u8]) -> Self {
        Buffer::Static(data)
    }

    /// Create an empty buffer.
    #[inline]
    pub const fn empty() -> Self {
        Buffer::Empty
    }

    /// Get the buffer contents as a byte slice.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        match self {
            Buffer::Mmap(mmap) => mmap.as_ref(),
            Buffer::Bytes(bytes) => bytes.as_ref(),
            Buffer::Static(data) => data,
            Buffer::Empty => &[],
        }
    }

    /// Get the buffer contents as a string slice.
    ///
    /// # Errors
    /// Returns an error if the buffer contains invalid UTF-8.
    #[inline]
    pub fn as_str(&self) -> Result<&str, Utf8Error> {
        std::str::from_utf8(self.as_slice())
    }

    /// Get the length of the buffer in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    /// Check if the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Create a view into this buffer without copying.
    ///
    /// # Panics
    /// Panics if the range is out of bounds.
    #[inline]
    pub fn view(&self, range: Range<usize>) -> BufferView<'_> {
        debug_assert!(range.end <= self.len(), "range out of bounds");
        BufferView {
            buffer: self,
            start: range.start,
            end: range.end,
        }
    }

    /// Create a view of the entire buffer.
    #[inline]
    pub fn view_all(&self) -> BufferView<'_> {
        BufferView {
            buffer: self,
            start: 0,
            end: self.len(),
        }
    }

    /// Slice the underlying bytes (for Bytes variant).
    /// For other variants, this creates a new Bytes from the slice.
    pub fn slice(&self, range: Range<usize>) -> Buffer {
        match self {
            Buffer::Bytes(bytes) => Buffer::Bytes(bytes.slice(range)),
            Buffer::Static(data) => Buffer::Static(&data[range]),
            Buffer::Mmap(_) | Buffer::Empty => {
                // For mmap, we need to copy to get an owned slice
                Buffer::Bytes(Bytes::copy_from_slice(&self.as_slice()[range]))
            }
        }
    }
}

impl Default for Buffer {
    #[inline]
    fn default() -> Self {
        Buffer::Empty
    }
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Buffer::Mmap(_) => f.debug_tuple("Buffer::Mmap").field(&self.len()).finish(),
            Buffer::Bytes(b) => f.debug_tuple("Buffer::Bytes").field(&b.len()).finish(),
            Buffer::Static(s) => f.debug_tuple("Buffer::Static").field(&s.len()).finish(),
            Buffer::Empty => f.debug_tuple("Buffer::Empty").finish(),
        }
    }
}

impl From<Vec<u8>> for Buffer {
    #[inline]
    fn from(vec: Vec<u8>) -> Self {
        Buffer::Bytes(Bytes::from(vec))
    }
}

impl From<String> for Buffer {
    #[inline]
    fn from(s: String) -> Self {
        Buffer::Bytes(Bytes::from(s))
    }
}

impl From<&'static str> for Buffer {
    #[inline]
    fn from(s: &'static str) -> Self {
        Buffer::Static(s.as_bytes())
    }
}

impl From<Bytes> for Buffer {
    #[inline]
    fn from(bytes: Bytes) -> Self {
        Buffer::Bytes(bytes)
    }
}

/// Zero-copy view into a [`Buffer`].
///
/// This is a lightweight reference type that points to a range within
/// a buffer without copying any data. Multiple views can reference
/// different ranges of the same buffer.
#[derive(Clone)]
pub struct BufferView<'a> {
    buffer: &'a Buffer,
    start: usize,
    end: usize,
}

impl<'a> BufferView<'a> {
    /// Create a new buffer view.
    #[inline]
    pub const fn new(buffer: &'a Buffer, start: usize, end: usize) -> Self {
        Self { buffer, start, end }
    }

    /// Create from a range.
    #[inline]
    pub fn from_range(buffer: &'a Buffer, range: Range<usize>) -> Self {
        Self {
            buffer,
            start: range.start,
            end: range.end,
        }
    }

    /// Get the view contents as a byte slice.
    #[inline]
    pub fn as_slice(&self) -> &'a [u8] {
        &self.buffer.as_slice()[self.start..self.end]
    }

    /// Get the view contents as a string slice.
    ///
    /// # Errors
    /// Returns an error if the view contains invalid UTF-8.
    #[inline]
    pub fn as_str(&self) -> Result<&'a str, Utf8Error> {
        std::str::from_utf8(self.as_slice())
    }

    /// Get the length of the view in bytes.
    #[inline]
    pub const fn len(&self) -> usize {
        self.end - self.start
    }

    /// Check if the view is empty.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.start == self.end
    }

    /// Get the start offset of this view within the buffer.
    #[inline]
    pub const fn start(&self) -> usize {
        self.start
    }

    /// Get the end offset of this view within the buffer.
    #[inline]
    pub const fn end(&self) -> usize {
        self.end
    }

    /// Get the range of this view within the buffer.
    #[inline]
    pub fn range(&self) -> Range<usize> {
        self.start..self.end
    }

    /// Create a sub-view relative to this view.
    ///
    /// # Panics
    /// Panics if the range is out of bounds.
    #[inline]
    pub fn subview(&self, range: Range<usize>) -> BufferView<'a> {
        debug_assert!(range.end <= self.len(), "subview range out of bounds");
        BufferView {
            buffer: self.buffer,
            start: self.start + range.start,
            end: self.start + range.end,
        }
    }
}

// Manual Copy impl since we use usize instead of Range
impl<'a> Copy for BufferView<'a> {}

impl<'a> std::fmt::Debug for BufferView<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferView")
            .field("start", &self.start)
            .field("end", &self.end)
            .field("len", &self.len())
            .finish()
    }
}

impl<'a> AsRef<[u8]> for BufferView<'a> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_from_bytes() {
        let buffer = Buffer::from_bytes(b"hello world".to_vec());
        assert_eq!(buffer.as_slice(), b"hello world");
        assert_eq!(buffer.as_str().unwrap(), "hello world");
        assert_eq!(buffer.len(), 11);
        assert!(!buffer.is_empty());
    }

    #[test]
    fn test_buffer_from_static() {
        let buffer = Buffer::from_static(b"static data");
        assert_eq!(buffer.as_slice(), b"static data");
        assert_eq!(buffer.len(), 11);
    }

    #[test]
    fn test_buffer_empty() {
        let buffer = Buffer::empty();
        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
        let empty: &[u8] = &[];
        assert_eq!(buffer.as_slice(), empty);
    }

    #[test]
    fn test_buffer_view() {
        let buffer = Buffer::from_bytes(b"hello world".to_vec());
        let view = buffer.view(0..5);
        assert_eq!(view.as_slice(), b"hello");
        assert_eq!(view.as_str().unwrap(), "hello");
        assert_eq!(view.len(), 5);
    }

    #[test]
    fn test_buffer_view_subview() {
        let buffer = Buffer::from_bytes(b"hello world".to_vec());
        let view = buffer.view(0..11);
        let subview = view.subview(6..11);
        assert_eq!(subview.as_str().unwrap(), "world");
    }

    #[test]
    fn test_buffer_slice() {
        let buffer = Buffer::from_bytes(b"hello world".to_vec());
        let sliced = buffer.slice(6..11);
        assert_eq!(sliced.as_str().unwrap(), "world");
    }

    #[test]
    fn test_buffer_from_string() {
        let buffer: Buffer = "test string".to_string().into();
        assert_eq!(buffer.as_str().unwrap(), "test string");
    }

    #[test]
    fn test_buffer_from_static_str() {
        let buffer: Buffer = "static".into();
        assert_eq!(buffer.as_str().unwrap(), "static");
    }

    #[test]
    fn test_buffer_clone() {
        let buffer = Buffer::from_bytes(b"clone me".to_vec());
        let cloned = buffer.clone();
        assert_eq!(buffer.as_slice(), cloned.as_slice());
    }

    #[test]
    fn test_buffer_view_range() {
        let buffer = Buffer::from_bytes(b"0123456789".to_vec());
        let view = buffer.view(3..7);
        assert_eq!(view.start(), 3);
        assert_eq!(view.end(), 7);
        assert_eq!(view.range(), 3..7);
    }
}
