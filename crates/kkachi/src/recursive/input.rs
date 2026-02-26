// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Multimodal input types for LLM interactions.
//!
//! Supports text, images, and audio as inputs to LLM modules.
//! Uses zero-copy `Cow` types for borrowed data and `SmallVec`
//! for inline storage of small multimodal payloads.
//!
//! # Examples
//!
//! ```
//! use kkachi::recursive::input::{Input, ContentType};
//!
//! // Simple text input (zero allocation for borrowed strings)
//! let text_input = Input::text("Describe this image");
//!
//! // Multimodal input with text and image
//! let multi_input = Input::multi()
//!     .text("What is in this image?")
//!     .image_png(b"\x89PNG\r\n\x1a\n")
//!     .build();
//! ```

use std::borrow::Cow;
use std::future::Future;

use smallvec::SmallVec;

use crate::error::Result;
use crate::recursive::llm::{Llm, LmOutput};

/// Content type for binary data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ContentType {
    /// Plain text content.
    Text = 0,
    /// PNG image.
    ImagePng = 1,
    /// JPEG image.
    ImageJpeg = 2,
    /// WebP image.
    ImageWebp = 3,
    /// WAV audio.
    AudioWav = 4,
    /// MP3 audio.
    AudioMp3 = 5,
}

impl ContentType {
    /// Get the MIME type string for this content type.
    pub fn mime_type(&self) -> &'static str {
        match self {
            ContentType::Text => "text/plain",
            ContentType::ImagePng => "image/png",
            ContentType::ImageJpeg => "image/jpeg",
            ContentType::ImageWebp => "image/webp",
            ContentType::AudioWav => "audio/wav",
            ContentType::AudioMp3 => "audio/mpeg",
        }
    }

    /// Check if this is an image content type.
    pub fn is_image(&self) -> bool {
        matches!(
            self,
            ContentType::ImagePng | ContentType::ImageJpeg | ContentType::ImageWebp
        )
    }

    /// Check if this is an audio content type.
    pub fn is_audio(&self) -> bool {
        matches!(self, ContentType::AudioWav | ContentType::AudioMp3)
    }
}

/// A single part of a multimodal input.
#[derive(Debug, Clone)]
pub enum InputPart<'a> {
    /// Text content (zero-copy when borrowed).
    Text(Cow<'a, str>),
    /// Binary content (image, audio, etc.).
    Binary {
        /// The content type of the binary data.
        content_type: ContentType,
        /// The raw binary data (zero-copy when borrowed).
        data: Cow<'a, [u8]>,
    },
}

impl<'a> InputPart<'a> {
    /// Get the text content if this is a text part.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            InputPart::Text(s) => Some(s),
            InputPart::Binary { .. } => None,
        }
    }

    /// Get the binary data and content type if this is a binary part.
    pub fn as_binary(&self) -> Option<(ContentType, &[u8])> {
        match self {
            InputPart::Text(_) => None,
            InputPart::Binary { content_type, data } => Some((*content_type, data)),
        }
    }

    /// Check if this part is a text part.
    pub fn is_text(&self) -> bool {
        matches!(self, InputPart::Text(_))
    }

    /// Check if this part is a binary part.
    pub fn is_binary(&self) -> bool {
        matches!(self, InputPart::Binary { .. })
    }

    /// Convert to an owned version with `'static` lifetime.
    pub fn into_owned(self) -> InputPart<'static> {
        match self {
            InputPart::Text(s) => InputPart::Text(Cow::Owned(s.into_owned())),
            InputPart::Binary { content_type, data } => InputPart::Binary {
                content_type,
                data: Cow::Owned(data.into_owned()),
            },
        }
    }
}

/// Input that can be text-only or multimodal.
///
/// The common case (text-only) avoids allocation entirely when using borrowed
/// strings. Multimodal inputs use `SmallVec` with inline storage for up to 4
/// parts before spilling to the heap.
#[derive(Debug, Clone)]
pub enum Input<'a> {
    /// Simple text input (common case, zero alloc for borrowed).
    Text(Cow<'a, str>),
    /// Multimodal input with multiple parts.
    Multi(SmallVec<[InputPart<'a>; 4]>),
}

impl<'a> Input<'a> {
    /// Create a text-only input from a borrowed string.
    pub fn text(s: &'a str) -> Self {
        Input::Text(Cow::Borrowed(s))
    }

    /// Create a text-only input from an owned string.
    pub fn text_owned(s: String) -> Self {
        Input::Text(Cow::Owned(s))
    }

    /// Create a multimodal input builder.
    pub fn multi() -> MultiInputBuilder<'a> {
        MultiInputBuilder {
            parts: SmallVec::new(),
        }
    }

    /// Get text content if this is a text-only input.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Input::Text(s) => Some(s),
            Input::Multi(_) => None,
        }
    }

    /// Check if this input is text-only.
    pub fn is_text(&self) -> bool {
        matches!(self, Input::Text(_))
    }

    /// Check if this input is multimodal.
    pub fn is_multi(&self) -> bool {
        matches!(self, Input::Multi(_))
    }

    /// Get the number of parts in this input.
    pub fn part_count(&self) -> usize {
        match self {
            Input::Text(_) => 1,
            Input::Multi(parts) => parts.len(),
        }
    }

    /// Get all parts (wraps text in a single-element iter if text-only).
    pub fn parts(&self) -> InputParts<'a, '_> {
        match self {
            Input::Text(s) => InputParts::Single(std::iter::once(InputPart::Text(s.clone()))),
            Input::Multi(parts) => InputParts::Multi(parts.iter()),
        }
    }

    /// Convert to owned version with `'static` lifetime.
    pub fn into_owned(self) -> Input<'static> {
        match self {
            Input::Text(s) => Input::Text(Cow::Owned(s.into_owned())),
            Input::Multi(parts) => {
                let owned: SmallVec<[InputPart<'static>; 4]> =
                    parts.into_iter().map(|p| p.into_owned()).collect();
                Input::Multi(owned)
            }
        }
    }

    /// Extract all text content from this input, concatenated.
    ///
    /// For text-only inputs, returns the text directly. For multimodal inputs,
    /// concatenates all text parts with newlines.
    pub fn all_text(&self) -> Cow<'_, str> {
        match self {
            Input::Text(s) => Cow::Borrowed(s),
            Input::Multi(parts) => {
                let mut result = String::new();
                for part in parts {
                    if let InputPart::Text(s) = part {
                        if !result.is_empty() {
                            result.push('\n');
                        }
                        result.push_str(s);
                    }
                }
                Cow::Owned(result)
            }
        }
    }
}

/// Builder for multimodal inputs.
///
/// Uses method chaining to construct a multimodal input with
/// multiple text and binary parts.
pub struct MultiInputBuilder<'a> {
    parts: SmallVec<[InputPart<'a>; 4]>,
}

impl<'a> MultiInputBuilder<'a> {
    /// Add a text part from a borrowed string.
    pub fn text(mut self, s: &'a str) -> Self {
        self.parts.push(InputPart::Text(Cow::Borrowed(s)));
        self
    }

    /// Add a text part from an owned string.
    pub fn text_owned(mut self, s: String) -> Self {
        self.parts.push(InputPart::Text(Cow::Owned(s)));
        self
    }

    /// Add a PNG image part.
    pub fn image_png(mut self, data: &'a [u8]) -> Self {
        self.parts.push(InputPart::Binary {
            content_type: ContentType::ImagePng,
            data: Cow::Borrowed(data),
        });
        self
    }

    /// Add a JPEG image part.
    pub fn image_jpeg(mut self, data: &'a [u8]) -> Self {
        self.parts.push(InputPart::Binary {
            content_type: ContentType::ImageJpeg,
            data: Cow::Borrowed(data),
        });
        self
    }

    /// Add a WebP image part.
    pub fn image_webp(mut self, data: &'a [u8]) -> Self {
        self.parts.push(InputPart::Binary {
            content_type: ContentType::ImageWebp,
            data: Cow::Borrowed(data),
        });
        self
    }

    /// Add a WAV audio part.
    pub fn audio_wav(mut self, data: &'a [u8]) -> Self {
        self.parts.push(InputPart::Binary {
            content_type: ContentType::AudioWav,
            data: Cow::Borrowed(data),
        });
        self
    }

    /// Add a MP3 audio part.
    pub fn audio_mp3(mut self, data: &'a [u8]) -> Self {
        self.parts.push(InputPart::Binary {
            content_type: ContentType::AudioMp3,
            data: Cow::Borrowed(data),
        });
        self
    }

    /// Add a binary part with an explicit content type.
    pub fn binary(mut self, content_type: ContentType, data: &'a [u8]) -> Self {
        self.parts.push(InputPart::Binary {
            content_type,
            data: Cow::Borrowed(data),
        });
        self
    }

    /// Add a binary part from owned data.
    pub fn binary_owned(mut self, content_type: ContentType, data: Vec<u8>) -> Self {
        self.parts.push(InputPart::Binary {
            content_type,
            data: Cow::Owned(data),
        });
        self
    }

    /// Build the multimodal input.
    pub fn build(self) -> Input<'a> {
        Input::Multi(self.parts)
    }
}

/// Iterator over input parts.
///
/// For text-only inputs, yields a single `InputPart::Text`. For multimodal
/// inputs, yields each part in order.
pub enum InputParts<'a, 'b> {
    /// Single text part (from `Input::Text`).
    Single(std::iter::Once<InputPart<'a>>),
    /// Multiple parts (from `Input::Multi`).
    Multi(std::slice::Iter<'b, InputPart<'a>>),
}

impl<'a, 'b> Iterator for InputParts<'a, 'b> {
    type Item = InputPart<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            InputParts::Single(iter) => iter.next(),
            InputParts::Multi(iter) => iter.next().cloned(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            InputParts::Single(iter) => iter.size_hint(),
            InputParts::Multi(iter) => iter.size_hint(),
        }
    }
}

impl<'a, 'b> ExactSizeIterator for InputParts<'a, 'b> {}

/// Extension trait for LLMs that support multimodal input.
///
/// This extends the base [`Llm`] trait with the ability to process
/// multimodal inputs (text, images, audio). Implementations should
/// handle each content type appropriately for the target LLM API.
///
/// # Examples
///
/// ```ignore
/// use kkachi::recursive::input::{Input, MultimodalLlm};
///
/// async fn describe_image<L: MultimodalLlm>(llm: &L, image_data: &[u8]) {
///     let input = Input::multi()
///         .text("Describe this image in detail.")
///         .image_png(image_data)
///         .build();
///
///     let output = llm.generate_multi(&input, "", None).await.unwrap();
///     println!("Description: {}", output.text);
/// }
/// ```
pub trait MultimodalLlm: Llm {
    /// The future type returned by `generate_multi()`.
    type MultiGenerateFut<'a>: Future<Output = Result<LmOutput>> + Send + 'a
    where
        Self: 'a;

    /// Generate a response from a multimodal input.
    ///
    /// # Arguments
    ///
    /// * `input` - The multimodal input (text, images, audio)
    /// * `context` - Additional context (e.g., from RAG)
    /// * `feedback` - Optional feedback from a previous iteration
    fn generate_multi<'a>(
        &'a self,
        input: &'a Input<'a>,
        context: &'a str,
        feedback: Option<&'a str>,
    ) -> Self::MultiGenerateFut<'a>;

    /// Check if this LLM supports a given content type.
    ///
    /// Implementations can override this to advertise which content
    /// types they accept. The default returns `true` for text only.
    fn supports_content_type(&self, content_type: ContentType) -> bool {
        matches!(content_type, ContentType::Text)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // ContentType tests
    // ========================================================================

    #[test]
    fn test_content_type_mime() {
        assert_eq!(ContentType::Text.mime_type(), "text/plain");
        assert_eq!(ContentType::ImagePng.mime_type(), "image/png");
        assert_eq!(ContentType::ImageJpeg.mime_type(), "image/jpeg");
        assert_eq!(ContentType::ImageWebp.mime_type(), "image/webp");
        assert_eq!(ContentType::AudioWav.mime_type(), "audio/wav");
        assert_eq!(ContentType::AudioMp3.mime_type(), "audio/mpeg");
    }

    #[test]
    fn test_content_type_is_image() {
        assert!(!ContentType::Text.is_image());
        assert!(ContentType::ImagePng.is_image());
        assert!(ContentType::ImageJpeg.is_image());
        assert!(ContentType::ImageWebp.is_image());
        assert!(!ContentType::AudioWav.is_image());
        assert!(!ContentType::AudioMp3.is_image());
    }

    #[test]
    fn test_content_type_is_audio() {
        assert!(!ContentType::Text.is_audio());
        assert!(!ContentType::ImagePng.is_audio());
        assert!(ContentType::AudioWav.is_audio());
        assert!(ContentType::AudioMp3.is_audio());
    }

    #[test]
    fn test_content_type_repr() {
        assert_eq!(ContentType::Text as u8, 0);
        assert_eq!(ContentType::ImagePng as u8, 1);
        assert_eq!(ContentType::ImageJpeg as u8, 2);
        assert_eq!(ContentType::ImageWebp as u8, 3);
        assert_eq!(ContentType::AudioWav as u8, 4);
        assert_eq!(ContentType::AudioMp3 as u8, 5);
    }

    // ========================================================================
    // InputPart tests
    // ========================================================================

    #[test]
    fn test_input_part_text() {
        let part = InputPart::Text(Cow::Borrowed("hello"));
        assert!(part.is_text());
        assert!(!part.is_binary());
        assert_eq!(part.as_text(), Some("hello"));
        assert!(part.as_binary().is_none());
    }

    #[test]
    fn test_input_part_binary() {
        let data = b"PNG data";
        let part = InputPart::Binary {
            content_type: ContentType::ImagePng,
            data: Cow::Borrowed(data.as_slice()),
        };
        assert!(!part.is_text());
        assert!(part.is_binary());
        assert!(part.as_text().is_none());

        let (ct, bytes) = part.as_binary().unwrap();
        assert_eq!(ct, ContentType::ImagePng);
        assert_eq!(bytes, b"PNG data");
    }

    #[test]
    fn test_input_part_into_owned() {
        let text = String::from("borrowed text");
        let part = InputPart::Text(Cow::Borrowed(&text));
        let owned: InputPart<'static> = part.into_owned();
        assert_eq!(owned.as_text(), Some("borrowed text"));

        let data = vec![1u8, 2, 3];
        let part = InputPart::Binary {
            content_type: ContentType::ImageJpeg,
            data: Cow::Borrowed(&data),
        };
        let owned: InputPart<'static> = part.into_owned();
        let (ct, bytes) = owned.as_binary().unwrap();
        assert_eq!(ct, ContentType::ImageJpeg);
        assert_eq!(bytes, &[1, 2, 3]);
    }

    // ========================================================================
    // Input tests
    // ========================================================================

    #[test]
    fn test_input_text_borrowed() {
        let input = Input::text("hello world");
        assert!(input.is_text());
        assert!(!input.is_multi());
        assert_eq!(input.as_text(), Some("hello world"));
        assert_eq!(input.part_count(), 1);
    }

    #[test]
    fn test_input_text_owned() {
        let input = Input::text_owned("owned string".to_string());
        assert!(input.is_text());
        assert_eq!(input.as_text(), Some("owned string"));
    }

    #[test]
    fn test_input_multi_builder() {
        let png_data = b"\x89PNG";
        let input = Input::multi()
            .text("Describe this image")
            .image_png(png_data.as_slice())
            .build();

        assert!(input.is_multi());
        assert!(!input.is_text());
        assert!(input.as_text().is_none());
        assert_eq!(input.part_count(), 2);
    }

    #[test]
    fn test_input_multi_all_types() {
        let png = b"png";
        let jpeg = b"jpeg";
        let webp = b"webp";
        let wav = b"wav";
        let mp3 = b"mp3";

        let input = Input::multi()
            .text("prompt")
            .image_png(png.as_slice())
            .image_jpeg(jpeg.as_slice())
            .image_webp(webp.as_slice())
            .audio_wav(wav.as_slice())
            .audio_mp3(mp3.as_slice())
            .build();

        assert_eq!(input.part_count(), 6);
    }

    #[test]
    fn test_input_multi_binary_generic() {
        let data = b"custom data";
        let input = Input::multi()
            .binary(ContentType::AudioWav, data.as_slice())
            .build();

        assert_eq!(input.part_count(), 1);
        let part = input.parts().next().unwrap();
        let (ct, bytes) = part.as_binary().unwrap();
        assert_eq!(ct, ContentType::AudioWav);
        assert_eq!(bytes, b"custom data");
    }

    #[test]
    fn test_input_multi_binary_owned() {
        let data = vec![10u8, 20, 30];
        let input = Input::multi()
            .binary_owned(ContentType::ImagePng, data)
            .build();

        assert_eq!(input.part_count(), 1);
    }

    #[test]
    fn test_input_multi_text_owned() {
        let input = Input::multi()
            .text_owned("owned text".to_string())
            .build();

        let part = input.parts().next().unwrap();
        assert_eq!(part.as_text(), Some("owned text"));
    }

    #[test]
    fn test_input_into_owned_text() {
        let s = String::from("temporary");
        let input = Input::text(&s);
        let owned: Input<'static> = input.into_owned();
        assert_eq!(owned.as_text(), Some("temporary"));
    }

    #[test]
    fn test_input_into_owned_multi() {
        let text = String::from("prompt");
        let data = vec![1u8, 2, 3];

        let input = Input::multi()
            .text(&text)
            .binary(ContentType::ImagePng, &data)
            .build();

        let owned: Input<'static> = input.into_owned();
        assert!(owned.is_multi());
        assert_eq!(owned.part_count(), 2);
    }

    // ========================================================================
    // InputParts iterator tests
    // ========================================================================

    #[test]
    fn test_parts_from_text_input() {
        let input = Input::text("hello");
        let parts: Vec<_> = input.parts().collect();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].as_text(), Some("hello"));
    }

    #[test]
    fn test_parts_from_multi_input() {
        let png = b"img";
        let input = Input::multi()
            .text("first")
            .image_png(png.as_slice())
            .text("second")
            .build();

        let parts: Vec<_> = input.parts().collect();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].as_text(), Some("first"));
        assert!(parts[1].is_binary());
        assert_eq!(parts[2].as_text(), Some("second"));
    }

    #[test]
    fn test_parts_size_hint() {
        let input = Input::text("one part");
        let iter = input.parts();
        assert_eq!(iter.size_hint(), (1, Some(1)));

        let input = Input::multi()
            .text("a")
            .text("b")
            .text("c")
            .build();
        let iter = input.parts();
        assert_eq!(iter.size_hint(), (3, Some(3)));
    }

    // ========================================================================
    // all_text tests
    // ========================================================================

    #[test]
    fn test_all_text_from_text_input() {
        let input = Input::text("simple text");
        assert_eq!(input.all_text().as_ref(), "simple text");
    }

    #[test]
    fn test_all_text_from_multi_input() {
        let img = b"img";
        let input = Input::multi()
            .text("first line")
            .image_png(img.as_slice())
            .text("second line")
            .build();

        assert_eq!(input.all_text().as_ref(), "first line\nsecond line");
    }

    #[test]
    fn test_all_text_no_text_parts() {
        let img = b"img";
        let input = Input::multi().image_png(img.as_slice()).build();
        assert_eq!(input.all_text().as_ref(), "");
    }

    // ========================================================================
    // Clone / Debug tests
    // ========================================================================

    #[test]
    fn test_input_clone() {
        let input = Input::text("clonable");
        let cloned = input.clone();
        assert_eq!(cloned.as_text(), Some("clonable"));
    }

    #[test]
    fn test_input_debug() {
        let input = Input::text("debug me");
        let debug = format!("{:?}", input);
        assert!(debug.contains("debug me"));
    }

    #[test]
    fn test_content_type_eq() {
        assert_eq!(ContentType::ImagePng, ContentType::ImagePng);
        assert_ne!(ContentType::ImagePng, ContentType::ImageJpeg);
    }

    // ========================================================================
    // Edge case tests
    // ========================================================================

    #[test]
    fn test_empty_text_input() {
        let input = Input::text("");
        assert!(input.is_text());
        assert_eq!(input.as_text(), Some(""));
        assert_eq!(input.part_count(), 1);
    }

    #[test]
    fn test_empty_multi_input() {
        let input = Input::multi().build();
        assert!(input.is_multi());
        assert_eq!(input.part_count(), 0);
        assert_eq!(input.parts().count(), 0);
    }

    #[test]
    fn test_large_binary_data() {
        let data = vec![0xFFu8; 1024 * 1024]; // 1 MB
        let input = Input::multi()
            .binary_owned(ContentType::ImagePng, data.clone())
            .build();

        assert_eq!(input.part_count(), 1);
        let part = input.parts().next().unwrap();
        let (_, bytes) = part.as_binary().unwrap();
        assert_eq!(bytes.len(), 1024 * 1024);
    }

    #[test]
    fn test_multi_input_many_parts() {
        // Test that SmallVec spills to heap gracefully beyond 4 inline elements
        let mut builder = Input::multi();
        for i in 0..10 {
            builder = builder.text_owned(format!("part {}", i));
        }
        let input = builder.build();
        assert_eq!(input.part_count(), 10);

        let texts: Vec<_> = input
            .parts()
            .filter_map(|p| p.as_text().map(|s| s.to_string()))
            .collect();
        assert_eq!(texts.len(), 10);
        assert_eq!(texts[0], "part 0");
        assert_eq!(texts[9], "part 9");
    }
}
