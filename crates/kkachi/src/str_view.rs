// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Zero-copy string view for high-performance string handling
//!
//! Provides [`StrView`] - a truly zero-copy string reference that is:
//! - Copy (just pointer + length, 16 bytes on 64-bit)
//! - Send + Sync when the underlying data is
//! - Zero allocation for all operations

use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::Range;

/// Truly zero-copy string reference.
///
/// This is just a pointer and length (16 bytes on 64-bit systems),
/// making it extremely cheap to copy and pass around. Unlike `&str`,
/// it can be stored in structs without lifetime issues when the
/// underlying buffer is known to outlive the view.
///
/// # Safety
///
/// The underlying data must:
/// - Be valid UTF-8
/// - Remain valid for the lifetime `'a`
/// - Not be modified while the view exists
#[derive(Copy, Clone)]
pub struct StrView<'a> {
    ptr: *const u8,
    len: usize,
    _marker: PhantomData<&'a str>,
}

// SAFETY: StrView is just a pointer to immutable data with a lifetime.
// If the underlying data is Send+Sync (which &str is), so is StrView.
unsafe impl<'a> Send for StrView<'a> {}
unsafe impl<'a> Sync for StrView<'a> {}

impl<'a> StrView<'a> {
    /// Create a new string view from a string slice.
    #[inline(always)]
    pub const fn new(s: &'a str) -> Self {
        Self {
            ptr: s.as_ptr(),
            len: s.len(),
            _marker: PhantomData,
        }
    }

    /// Create an empty string view.
    #[inline(always)]
    pub const fn empty() -> Self {
        Self {
            ptr: std::ptr::NonNull::dangling().as_ptr(),
            len: 0,
            _marker: PhantomData,
        }
    }

    /// Create a string view from raw parts.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `ptr` points to valid UTF-8 data
    /// - The data is at least `len` bytes
    /// - The data remains valid for lifetime `'a`
    #[inline(always)]
    pub const unsafe fn from_raw_parts(ptr: *const u8, len: usize) -> Self {
        Self {
            ptr,
            len,
            _marker: PhantomData,
        }
    }

    /// Get the string slice this view represents.
    #[inline(always)]
    pub fn as_str(&self) -> &'a str {
        // SAFETY: We maintain the invariant that ptr/len point to valid UTF-8
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(self.ptr, self.len)) }
    }

    /// Get the underlying bytes.
    #[inline(always)]
    pub fn as_bytes(&self) -> &'a [u8] {
        // SAFETY: ptr/len are valid by construction
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Get the length in bytes.
    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Check if the view is empty.
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get a subview by byte range.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The range is out of bounds
    /// - The range doesn't lie on UTF-8 character boundaries
    #[inline]
    pub fn slice(&self, range: Range<usize>) -> StrView<'a> {
        let s = self.as_str();
        StrView::new(&s[range])
    }

    /// Try to get a subview by byte range.
    ///
    /// Returns `None` if the range is invalid or doesn't lie on
    /// UTF-8 character boundaries.
    #[inline]
    pub fn try_slice(&self, range: Range<usize>) -> Option<StrView<'a>> {
        if range.end > self.len {
            return None;
        }
        let s = self.as_str();
        s.get(range).map(StrView::new)
    }

    /// Get a subview by byte range without bounds checking.
    ///
    /// # Safety
    ///
    /// The range must be within bounds and on UTF-8 character boundaries.
    #[inline(always)]
    pub unsafe fn slice_unchecked(&self, range: Range<usize>) -> StrView<'a> {
        StrView {
            ptr: self.ptr.add(range.start),
            len: range.end - range.start,
            _marker: PhantomData,
        }
    }

    /// Split the view at the first occurrence of a character.
    #[inline]
    pub fn split_once(&self, c: char) -> Option<(StrView<'a>, StrView<'a>)> {
        let s = self.as_str();
        s.find(c).map(|pos| {
            let (before, after) = s.split_at(pos);
            (StrView::new(before), StrView::new(&after[c.len_utf8()..]))
        })
    }

    /// Strip a prefix from the view.
    #[inline]
    pub fn strip_prefix(&self, prefix: &str) -> Option<StrView<'a>> {
        self.as_str().strip_prefix(prefix).map(StrView::new)
    }

    /// Strip a suffix from the view.
    #[inline]
    pub fn strip_suffix(&self, suffix: &str) -> Option<StrView<'a>> {
        self.as_str().strip_suffix(suffix).map(StrView::new)
    }

    /// Trim whitespace from both ends.
    #[inline]
    pub fn trim(&self) -> StrView<'a> {
        StrView::new(self.as_str().trim())
    }

    /// Check if the view starts with a prefix.
    #[inline]
    pub fn starts_with(&self, prefix: &str) -> bool {
        self.as_str().starts_with(prefix)
    }

    /// Check if the view ends with a suffix.
    #[inline]
    pub fn ends_with(&self, suffix: &str) -> bool {
        self.as_str().ends_with(suffix)
    }

    /// Check if the view contains a pattern.
    #[inline]
    pub fn contains(&self, pattern: &str) -> bool {
        self.as_str().contains(pattern)
    }

    /// Find the first occurrence of a pattern.
    #[inline]
    pub fn find(&self, pattern: &str) -> Option<usize> {
        self.as_str().find(pattern)
    }

    /// Get the raw pointer to the string data.
    #[inline(always)]
    pub const fn as_ptr(&self) -> *const u8 {
        self.ptr
    }
}

impl<'a> Default for StrView<'a> {
    #[inline]
    fn default() -> Self {
        Self::empty()
    }
}

impl<'a> fmt::Debug for StrView<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_str(), f)
    }
}

impl<'a> fmt::Display for StrView<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_str(), f)
    }
}

impl<'a> PartialEq for StrView<'a> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}

impl<'a> Eq for StrView<'a> {}

impl<'a> PartialEq<str> for StrView<'a> {
    #[inline]
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl<'a> PartialEq<&str> for StrView<'a> {
    #[inline]
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl<'a> PartialEq<StrView<'a>> for str {
    #[inline]
    fn eq(&self, other: &StrView<'a>) -> bool {
        self == other.as_str()
    }
}

impl<'a> PartialOrd for StrView<'a> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a> Ord for StrView<'a> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_str().cmp(other.as_str())
    }
}

impl<'a> Hash for StrView<'a> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_str().hash(state)
    }
}

impl<'a> AsRef<str> for StrView<'a> {
    #[inline]
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl<'a> AsRef<[u8]> for StrView<'a> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl<'a> From<&'a str> for StrView<'a> {
    #[inline]
    fn from(s: &'a str) -> Self {
        Self::new(s)
    }
}

impl<'a> From<&'a String> for StrView<'a> {
    #[inline]
    fn from(s: &'a String) -> Self {
        Self::new(s.as_str())
    }
}

/// Iterator over lines in a [`StrView`].
pub struct Lines<'a> {
    remaining: StrView<'a>,
}

impl<'a> Iterator for Lines<'a> {
    type Item = StrView<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining.is_empty() {
            return None;
        }

        let s = self.remaining.as_str();
        match s.find('\n') {
            Some(pos) => {
                let line = &s[..pos];
                let rest = &s[pos + 1..];
                self.remaining = StrView::new(rest);
                // Handle \r\n
                let line = line.strip_suffix('\r').unwrap_or(line);
                Some(StrView::new(line))
            }
            None => {
                let line = self.remaining;
                self.remaining = StrView::empty();
                Some(line)
            }
        }
    }
}

impl<'a> StrView<'a> {
    /// Iterate over lines in the string view.
    #[inline]
    pub fn lines(&self) -> Lines<'a> {
        Lines { remaining: *self }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strview_new() {
        let s = "hello world";
        let view = StrView::new(s);
        assert_eq!(view.as_str(), "hello world");
        assert_eq!(view.len(), 11);
        assert!(!view.is_empty());
    }

    #[test]
    fn test_strview_empty() {
        let view = StrView::empty();
        assert_eq!(view.as_str(), "");
        assert_eq!(view.len(), 0);
        assert!(view.is_empty());
    }

    #[test]
    fn test_strview_slice() {
        let s = "hello world";
        let view = StrView::new(s);
        let sub = view.slice(0..5);
        assert_eq!(sub.as_str(), "hello");
    }

    #[test]
    fn test_strview_try_slice() {
        let view = StrView::new("hello");
        assert!(view.try_slice(0..5).is_some());
        assert!(view.try_slice(0..10).is_none());
    }

    #[test]
    fn test_strview_split_once() {
        let view = StrView::new("key=value");
        let (before, after) = view.split_once('=').unwrap();
        assert_eq!(before.as_str(), "key");
        assert_eq!(after.as_str(), "value");
    }

    #[test]
    fn test_strview_trim() {
        let view = StrView::new("  hello  ");
        assert_eq!(view.trim().as_str(), "hello");
    }

    #[test]
    fn test_strview_strip_prefix() {
        let view = StrView::new("prefix_value");
        let stripped = view.strip_prefix("prefix_").unwrap();
        assert_eq!(stripped.as_str(), "value");
    }

    #[test]
    fn test_strview_eq() {
        let view = StrView::new("hello");
        assert_eq!(view, "hello");
        assert!(view == "hello");
    }

    #[test]
    fn test_strview_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(StrView::new("hello"));
        assert!(set.contains(&StrView::new("hello")));
    }

    #[test]
    fn test_strview_ord() {
        let a = StrView::new("apple");
        let b = StrView::new("banana");
        assert!(a < b);
    }

    #[test]
    fn test_strview_lines() {
        let view = StrView::new("line1\nline2\nline3");
        let lines: Vec<_> = view.lines().collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0].as_str(), "line1");
        assert_eq!(lines[1].as_str(), "line2");
        assert_eq!(lines[2].as_str(), "line3");
    }

    #[test]
    fn test_strview_lines_crlf() {
        // Matches Rust's str::lines() behavior - trailing newline doesn't create empty line
        let view = StrView::new("line1\r\nline2\r\n");
        let lines: Vec<_> = view.lines().collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].as_str(), "line1");
        assert_eq!(lines[1].as_str(), "line2");
    }

    #[test]
    fn test_strview_copy() {
        let view = StrView::new("hello");
        let copy = view; // Copy, not move
        assert_eq!(view.as_str(), copy.as_str());
    }

    #[test]
    fn test_strview_size() {
        // StrView should be 16 bytes on 64-bit (ptr + len)
        assert_eq!(std::mem::size_of::<StrView>(), 16);
    }

    #[test]
    fn test_strview_from() {
        let s = String::from("owned");
        let view: StrView = (&s).into();
        assert_eq!(view.as_str(), "owned");

        let view2: StrView = "literal".into();
        assert_eq!(view2.as_str(), "literal");
    }

    #[test]
    fn test_strview_find() {
        let view = StrView::new("hello world");
        assert_eq!(view.find("world"), Some(6));
        assert_eq!(view.find("xyz"), None);
    }

    #[test]
    fn test_strview_contains() {
        let view = StrView::new("hello world");
        assert!(view.contains("world"));
        assert!(!view.contains("xyz"));
    }
}
