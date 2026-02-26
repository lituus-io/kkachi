// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Signature system for defining input/output contracts

use crate::error::{Error, Result};
use crate::field::{Field, FieldType};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

/// A signature defining the input/output contract of a module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signature<'a> {
    /// Instruction/prompt for the task
    #[serde(borrow)]
    pub instructions: Cow<'a, str>,

    /// Input fields
    #[serde(borrow)]
    pub input_fields: Vec<Field<'a>>,

    /// Output fields
    #[serde(borrow)]
    pub output_fields: Vec<Field<'a>>,
}

impl<'a> Signature<'a> {
    /// Create a new signature
    pub fn new(instructions: impl Into<Cow<'a, str>>) -> Self {
        Self {
            instructions: instructions.into(),
            input_fields: Vec::new(),
            output_fields: Vec::new(),
        }
    }

    /// Add an input field
    pub fn add_input(mut self, field: Field<'a>) -> Result<Self> {
        if field.field_type != FieldType::Input {
            return Err(Error::signature("Field must be an input field"));
        }
        self.input_fields.push(field);
        Ok(self)
    }

    /// Add an output field
    pub fn add_output(mut self, field: Field<'a>) -> Result<Self> {
        if field.field_type != FieldType::Output {
            return Err(Error::signature("Field must be an output field"));
        }
        self.output_fields.push(field);
        Ok(self)
    }

    /// Get all fields (inputs then outputs)
    pub fn all_fields(&self) -> impl Iterator<Item = &Field<'a>> {
        self.input_fields.iter().chain(self.output_fields.iter())
    }

    /// Convert to owned version
    pub fn into_owned(self) -> Signature<'static> {
        Signature {
            instructions: Cow::Owned(self.instructions.into_owned()),
            input_fields: self
                .input_fields
                .into_iter()
                .map(|f| f.into_owned())
                .collect(),
            output_fields: self
                .output_fields
                .into_iter()
                .map(|f| f.into_owned())
                .collect(),
        }
    }

    /// Parse a signature from string format: "input1, input2 -> output1, output2"
    pub fn parse(s: &'a str) -> Result<Self> {
        let parts: Vec<&str> = s.split("->").map(|p| p.trim()).collect();
        if parts.len() != 2 {
            return Err(Error::signature(
                "Signature must be in format 'inputs -> outputs'",
            ));
        }

        let mut sig = Self::new("");

        // Parse input fields
        for input in parts[0]
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
        {
            let field = Field::new(input, format!("Input: {}", input), FieldType::Input);
            sig.input_fields.push(field);
        }

        // Parse output fields
        for output in parts[1]
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
        {
            let field = Field::new(output, format!("Output: {}", output), FieldType::Output);
            sig.output_fields.push(field);
        }

        Ok(sig)
    }

    /// Convert signature to string format
    pub fn to_string_format(&self) -> String {
        let inputs: Vec<&str> = self.input_fields.iter().map(|f| f.name.as_ref()).collect();
        let outputs: Vec<&str> = self.output_fields.iter().map(|f| f.name.as_ref()).collect();
        format!("{} -> {}", inputs.join(", "), outputs.join(", "))
    }
}

/// Builder for creating signatures
pub struct SignatureBuilder<'a> {
    signature: Signature<'a>,
}

impl<'a> SignatureBuilder<'a> {
    /// Create a new builder
    pub fn new(instructions: impl Into<Cow<'a, str>>) -> Self {
        Self {
            signature: Signature::new(instructions),
        }
    }

    /// Add an input field
    pub fn input(mut self, field: Field<'a>) -> Result<Self> {
        self.signature = self.signature.add_input(field)?;
        Ok(self)
    }

    /// Add an output field
    pub fn output(mut self, field: Field<'a>) -> Result<Self> {
        self.signature = self.signature.add_output(field)?;
        Ok(self)
    }

    /// Build the signature
    pub fn build(self) -> Signature<'a> {
        self.signature
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::InputField;

    #[test]
    fn test_signature_from_str() {
        let sig = Signature::parse("question, context -> answer").unwrap();
        assert_eq!(sig.input_fields.len(), 2);
        assert_eq!(sig.output_fields.len(), 1);
        assert_eq!(sig.input_fields[0].name, "question");
        assert_eq!(sig.input_fields[1].name, "context");
        assert_eq!(sig.output_fields[0].name, "answer");
    }

    #[test]
    fn test_signature_to_string() {
        let sig = Signature::parse("q, ctx -> a").unwrap();
        assert_eq!(sig.to_string_format(), "q, ctx -> a");
    }

    #[test]
    fn test_signature_builder() {
        let sig = SignatureBuilder::new("Answer the question")
            .input(InputField::create("question", "User question"))
            .unwrap()
            .build();

        assert_eq!(sig.instructions, "Answer the question");
        assert_eq!(sig.input_fields.len(), 1);
    }
}
