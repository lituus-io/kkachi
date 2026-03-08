// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Composable module system for building complex LLM programs.
//!
//! Provides the [`ComposableModule`] trait for defining modules that can be
//! composed, optimized, and serialized. Uses GATs for zero-cost async.
//!
//! # Architecture
//!
//! The composable system consists of three layers:
//!
//! 1. **[`ComposableModule`]** - trait for modules with typed I/O, state
//!    save/load, and zero-cost async via GATs.
//! 2. **[`ModuleState`]** - a data-only, `Cow`-based snapshot of a module's
//!    learnable parameters (instruction, demos, children).
//! 3. **[`kkachi_module!`]** - declarative macro that auto-generates
//!    `save_state`/`load_state` by recursing into `#[module]` fields.
//!
//! # Examples
//!
//! ```
//! use kkachi::composable::ModuleState;
//!
//! let state = ModuleState::new("my_module")
//!     .with_instructions("Translate English to French.")
//!     .with_demo("Hello", "Bonjour")
//!     .with_demo("Goodbye", "Au revoir");
//!
//! assert_eq!(state.demos.len(), 2);
//! ```

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::borrow::Cow;
use std::future::Future;

// ---------------------------------------------------------------------------
// ComposableModule trait
// ---------------------------------------------------------------------------

/// Trait for composable LLM modules with typed I/O and state management.
///
/// Unlike the simpler [`Module`](crate::Module) trait, `ComposableModule`
/// provides:
/// - Typed, per-module `Input` and `Output` (not just `Inputs -> Prediction`)
/// - State serialization for saving/loading optimized parameters
/// - A stable `name()` for identifying modules in hierarchies
///
/// # GATs
///
/// The `ForwardFut` associated type uses Generic Associated Types so each
/// implementation can return its own future without boxing.
///
/// # Example
///
/// ```ignore
/// use kkachi::composable::{ComposableModule, ModuleState};
///
/// struct Echo;
///
/// impl ComposableModule for Echo {
///     type Input<'a> = &'a str;
///     type Output<'a> = String;
///     type ForwardFut<'a> = std::future::Ready<kkachi::error::Result<String>>;
///
///     fn forward<'a>(&'a self, input: &'a str) -> Self::ForwardFut<'a> {
///         std::future::ready(Ok(input.to_string()))
///     }
///
///     fn name(&self) -> &str { "echo" }
///
///     fn save_state(&self) -> ModuleState<'_> {
///         ModuleState::new(self.name())
///     }
///
///     fn load_state(&mut self, _state: &ModuleState<'_>) -> bool { true }
/// }
/// ```
pub trait ComposableModule: Send + Sync {
    /// Input type for this module. The lifetime allows borrowing from the caller.
    type Input<'a>
    where
        Self: 'a;

    /// Output type for this module. The lifetime allows borrowing from `self`.
    type Output<'a>
    where
        Self: 'a;

    /// Future returned by [`forward`](Self::forward).
    type ForwardFut<'a>: Future<Output = crate::error::Result<Self::Output<'a>>> + Send + 'a
    where
        Self: 'a;

    /// Execute the module on the given input.
    fn forward<'a>(&'a self, input: Self::Input<'a>) -> Self::ForwardFut<'a>;

    /// Human-readable name for this module (used in state trees and logging).
    fn name(&self) -> &str;

    /// Snapshot the module's learnable state (instruction, demos, children).
    fn save_state(&self) -> ModuleState<'_>;

    /// Restore learnable state from a snapshot.
    ///
    /// Returns `true` if the state was successfully applied, `false` if
    /// the state was incompatible (e.g., wrong module name).
    fn load_state(&mut self, state: &ModuleState<'_>) -> bool;
}

// ---------------------------------------------------------------------------
// ModuleState
// ---------------------------------------------------------------------------

/// Data-only snapshot of a module's learnable parameters.
///
/// `ModuleState` uses `Cow` throughout so it can borrow from a live module
/// (zero-copy `save_state`) or own data when loaded from disk.
///
/// # Serialization
///
/// `ModuleState` can be serialized to/from JSON for persistence. When
/// deserializing, all strings become owned (`'static`).
///
/// # Examples
///
/// ```
/// use kkachi::composable::ModuleState;
///
/// let state = ModuleState::new("predict")
///     .with_instructions("Answer the question.")
///     .with_demo("What is 2+2?", "4")
///     .with_child(ModuleState::new("retrieve"));
///
/// assert_eq!(state.name, "predict");
/// assert!(state.instructions.is_some());
/// assert_eq!(state.demos.len(), 1);
/// assert_eq!(state.children.len(), 1);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleState<'a> {
    /// Module name (matches [`ComposableModule::name`]).
    #[serde(borrow)]
    pub name: Cow<'a, str>,

    /// Optimized instruction/system prompt, if any.
    #[serde(borrow, default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<Cow<'a, str>>,

    /// Selected demonstration examples as (input, output) pairs.
    #[serde(borrow, default, skip_serializing_if = "SmallVec::is_empty")]
    pub demos: SmallVec<[(Cow<'a, str>, Cow<'a, str>); 4]>,

    /// Child module states (for hierarchical/composed modules).
    ///
    /// Uses `Box` for indirection to avoid infinite struct size.
    #[serde(borrow, default, skip_serializing_if = "SmallVec::is_empty")]
    pub children: SmallVec<[Box<ModuleState<'a>>; 4]>,
}

impl<'a> ModuleState<'a> {
    /// Create a new state with just a name.
    pub fn new(name: impl Into<Cow<'a, str>>) -> Self {
        Self {
            name: name.into(),
            instructions: None,
            demos: SmallVec::new(),
            children: SmallVec::new(),
        }
    }

    /// Set the optimized instruction, returning `self` for chaining.
    pub fn with_instructions(mut self, instructions: impl Into<Cow<'a, str>>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Add a demonstration example, returning `self` for chaining.
    pub fn with_demo(
        mut self,
        input: impl Into<Cow<'a, str>>,
        output: impl Into<Cow<'a, str>>,
    ) -> Self {
        self.demos.push((input.into(), output.into()));
        self
    }

    /// Add a child module state, returning `self` for chaining.
    pub fn with_child(mut self, child: ModuleState<'a>) -> Self {
        self.children.push(Box::new(child));
        self
    }

    /// Convert all borrowed data to owned, producing a `'static` state.
    ///
    /// This is necessary before storing a state long-term or sending it
    /// across thread boundaries where the original module may be dropped.
    pub fn into_owned(self) -> ModuleState<'static> {
        ModuleState {
            name: Cow::Owned(self.name.into_owned()),
            instructions: self.instructions.map(|s| Cow::Owned(s.into_owned())),
            demos: self
                .demos
                .into_iter()
                .map(|(i, o)| (Cow::Owned(i.into_owned()), Cow::Owned(o.into_owned())))
                .collect(),
            children: self
                .children
                .into_iter()
                .map(|c| Box::new((*c).into_owned()))
                .collect(),
        }
    }

    /// Find a child state by name (non-recursive).
    pub fn child(&self, name: &str) -> Option<&ModuleState<'a>> {
        self.children
            .iter()
            .find(|c| c.name == name)
            .map(|c| c.as_ref())
    }

    /// Find a child state by name (non-recursive, mutable).
    pub fn child_mut(&mut self, name: &str) -> Option<&mut ModuleState<'a>> {
        self.children
            .iter_mut()
            .find(|c| c.name == name)
            .map(|c| c.as_mut())
    }

    /// Serialize this state to a JSON string.
    pub fn to_json(&self) -> crate::error::Result<String> {
        let json = serde_json::to_string_pretty(self)?;
        Ok(json)
    }

    /// Deserialize a state from a JSON string (produces owned `'static` data).
    pub fn from_json(json: &str) -> crate::error::Result<ModuleState<'static>> {
        let state: ModuleState<'_> = serde_json::from_str(json)?;
        Ok(state.into_owned())
    }
}

// ---------------------------------------------------------------------------
// kkachi_module! macro
// ---------------------------------------------------------------------------

/// Declarative macro for auto-generating `save_state` / `load_state`.
///
/// Fields annotated with `#[module]` are treated as sub-modules whose state
/// is recursively saved and restored. All other fields are ignored by the
/// state machinery.
///
/// # Usage
///
/// ```ignore
/// use kkachi::kkachi_module;
///
/// kkachi_module! {
///     pub struct RAG<'a, L: Llm> {
///         #[module] retrieve: MemoryModule<'a>,
///         #[module] predict: ReasonModule<'a, L, Checks>,
///         k: usize,
///     }
/// }
/// ```
///
/// This expands roughly to:
///
/// ```ignore
/// pub struct RAG<'a, L: Llm> {
///     pub retrieve: MemoryModule<'a>,
///     pub predict: ReasonModule<'a, L, Checks>,
///     pub k: usize,
/// }
///
/// impl<'a, L: Llm> RAG<'a, L> {
///     fn save_composable_state(&self) -> ModuleState<'_> { ... }
///     fn load_composable_state(&mut self, state: &ModuleState<'_>) -> bool { ... }
/// }
/// ```
/// Declarative macro for composable module structs.
///
/// Generates `save_composable_state` and `load_composable_state` methods
/// that recurse into `#[module]`-annotated fields.
///
/// Fields marked `#[module]` must implement `save_composable_state` and
/// `load_composable_state` (either manually or via this macro). Other
/// fields are left untouched during state save/load.
#[macro_export]
macro_rules! kkachi_module {
    // Main entry: parse the struct, then dispatch to the TT muncher
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident $(<$($gen:tt),*>)? {
            $($body:tt)*
        }
    ) => {
        // Emit the struct definition via the field collector
        $crate::kkachi_module!(@collect_struct
            meta = [$(#[$meta])*],
            vis = [$vis],
            name = $name,
            gen = [$($($gen),*)?],
            fields = [],
            module_fields = [],
            rest = [$($body)*]
        );
    };

    // -----------------------------------------------------------------------
    // TT muncher: collect fields from the struct body
    // -----------------------------------------------------------------------

    // Match a #[module] field
    (@collect_struct
        meta = [$($meta:meta)*],
        vis = [$vis:vis],
        name = $name:ident,
        gen = [$($gen:tt)*],
        fields = [$($fields:tt)*],
        module_fields = [$($mfields:tt)*],
        rest = [#[module] $fvis:vis $field:ident : $fty:ty , $($rest:tt)*]
    ) => {
        $crate::kkachi_module!(@collect_struct
            meta = [$($meta)*],
            vis = [$vis],
            name = $name,
            gen = [$($gen)*],
            fields = [$($fields)* $fvis $field : $fty ,],
            module_fields = [$($mfields)* $field ,],
            rest = [$($rest)*]
        );
    };

    // Match a #[module] field (trailing, no comma)
    (@collect_struct
        meta = [$($meta:meta)*],
        vis = [$vis:vis],
        name = $name:ident,
        gen = [$($gen:tt)*],
        fields = [$($fields:tt)*],
        module_fields = [$($mfields:tt)*],
        rest = [#[module] $fvis:vis $field:ident : $fty:ty]
    ) => {
        $crate::kkachi_module!(@collect_struct
            meta = [$($meta)*],
            vis = [$vis],
            name = $name,
            gen = [$($gen)*],
            fields = [$($fields)* $fvis $field : $fty ,],
            module_fields = [$($mfields)* $field ,],
            rest = []
        );
    };

    // Match a regular field (no #[module])
    (@collect_struct
        meta = [$($meta:meta)*],
        vis = [$vis:vis],
        name = $name:ident,
        gen = [$($gen:tt)*],
        fields = [$($fields:tt)*],
        module_fields = [$($mfields:tt)*],
        rest = [$fvis:vis $field:ident : $fty:ty , $($rest:tt)*]
    ) => {
        $crate::kkachi_module!(@collect_struct
            meta = [$($meta)*],
            vis = [$vis],
            name = $name,
            gen = [$($gen)*],
            fields = [$($fields)* $fvis $field : $fty ,],
            module_fields = [$($mfields)*],
            rest = [$($rest)*]
        );
    };

    // Match a regular field (trailing, no comma)
    (@collect_struct
        meta = [$($meta:meta)*],
        vis = [$vis:vis],
        name = $name:ident,
        gen = [$($gen:tt)*],
        fields = [$($fields:tt)*],
        module_fields = [$($mfields:tt)*],
        rest = [$fvis:vis $field:ident : $fty:ty]
    ) => {
        $crate::kkachi_module!(@collect_struct
            meta = [$($meta)*],
            vis = [$vis],
            name = $name,
            gen = [$($gen)*],
            fields = [$($fields)* $fvis $field : $fty ,],
            module_fields = [$($mfields)*],
            rest = []
        );
    };

    // Terminal: all fields collected, emit struct + impl
    (@collect_struct
        meta = [$(#[$meta:meta])*],
        vis = [$vis:vis],
        name = $name:ident,
        gen = [$($gen:tt)*],
        fields = [$($fvis:vis $field:ident : $fty:ty ,)*],
        module_fields = [$($mfield:ident ,)*],
        rest = []
    ) => {
        $(#[$meta])*
        $vis struct $name $(<$($gen)*>)? {
            $($fvis $field : $fty ,)*
        }

        impl $(<$($gen)*>)? $name $(<$($gen)*>)? {
            /// Save the composable state of all `#[module]` children.
            #[allow(unused_mut)]
            pub fn save_composable_state(&self) -> $crate::composable::ModuleState<'_> {
                let mut state = $crate::composable::ModuleState::new(stringify!($name));
                $(
                    state.children.push(
                        Box::new(self.$mfield.save_composable_state())
                    );
                )*
                state
            }

            /// Load composable state into all `#[module]` children.
            ///
            /// Returns `true` if the top-level name matched and all children
            /// loaded successfully.
            #[allow(unused_variables, unused_mut)]
            pub fn load_composable_state(
                &mut self,
                state: &$crate::composable::ModuleState<'_>,
            ) -> bool {
                if state.name != stringify!($name) {
                    return false;
                }
                let mut ok = true;
                $(
                    if let Some(child) = state.child(stringify!($mfield)) {
                        if !self.$mfield.load_composable_state(child) {
                            ok = false;
                        }
                    }
                )*
                ok
            }
        }
    };
}

// ---------------------------------------------------------------------------
// PredictModule
// ---------------------------------------------------------------------------

/// A simple composable module that wraps an LLM call with optional demos.
///
/// `PredictModule` implements [`ComposableModule`] with:
/// - **Input**: `&'a str` (the prompt)
/// - **Output**: `String` (the LLM response)
///
/// It stores an instruction, optional validator, and few-shot demos that
/// are prepended to the prompt.
///
/// # Examples
///
/// ```ignore
/// use kkachi::composable::PredictModule;
/// use kkachi::recursive::{MockLlm, NoValidation};
///
/// let llm = MockLlm::new(|p, _| format!("response to: {}", p));
/// let module = PredictModule::new("predict", &llm, "Answer the question.");
///
/// // Use as a ComposableModule
/// let output = module.forward("What is 2+2?").await.unwrap();
/// assert!(output.contains("response to:"));
/// ```
#[allow(dead_code)]
pub struct PredictModule<
    'a,
    L: crate::recursive::Llm,
    V: crate::recursive::Validate = crate::recursive::NoValidation,
> {
    name: &'a str,
    llm: &'a L,
    instruction: String,
    validator: V,
    demos: SmallVec<[(String, String); 4]>,
}

impl<'a, L: crate::recursive::Llm> PredictModule<'a, L, crate::recursive::NoValidation> {
    /// Create a new predict module with no validator.
    pub fn new(name: &'a str, llm: &'a L, instruction: impl Into<String>) -> Self {
        Self {
            name,
            llm,
            instruction: instruction.into(),
            validator: crate::recursive::NoValidation,
            demos: SmallVec::new(),
        }
    }
}

impl<'a, L: crate::recursive::Llm, V: crate::recursive::Validate> PredictModule<'a, L, V> {
    /// Set a validator, returning a new module with the validator type changed.
    pub fn validate<V2: crate::recursive::Validate>(
        self,
        validator: V2,
    ) -> PredictModule<'a, L, V2> {
        PredictModule {
            name: self.name,
            llm: self.llm,
            instruction: self.instruction,
            validator,
            demos: self.demos,
        }
    }

    /// Add a few-shot demonstration example.
    pub fn demo(mut self, input: impl Into<String>, output: impl Into<String>) -> Self {
        self.demos.push((input.into(), output.into()));
        self
    }

    /// Set the instruction.
    pub fn instruction(mut self, instruction: impl Into<String>) -> Self {
        self.instruction = instruction.into();
        self
    }

    /// Build the full prompt including instruction, demos, and user input.
    fn build_prompt(&self, input: &str) -> String {
        let mut prompt = String::with_capacity(
            self.instruction.len() + self.demos.len() * 64 + input.len() + 32,
        );

        prompt.push_str(&self.instruction);
        prompt.push('\n');

        for (demo_in, demo_out) in &self.demos {
            prompt.push_str("\nInput: ");
            prompt.push_str(demo_in);
            prompt.push_str("\nOutput: ");
            prompt.push_str(demo_out);
            prompt.push('\n');
        }

        prompt.push_str("\nInput: ");
        prompt.push_str(input);
        prompt.push_str("\nOutput:");

        prompt
    }

    /// Save composable state (instruction + demos).
    pub fn save_composable_state(&self) -> ModuleState<'_> {
        let mut state =
            ModuleState::new(self.name).with_instructions(Cow::Borrowed(self.instruction.as_str()));

        for (demo_in, demo_out) in &self.demos {
            state.demos.push((
                Cow::Borrowed(demo_in.as_str()),
                Cow::Borrowed(demo_out.as_str()),
            ));
        }

        state
    }

    /// Load composable state (instruction + demos).
    ///
    /// Returns `true` if the state name matched and was applied.
    pub fn load_composable_state(&mut self, state: &ModuleState<'_>) -> bool {
        if state.name != self.name {
            return false;
        }

        if let Some(ref instr) = state.instructions {
            self.instruction = instr.to_string();
        }

        self.demos.clear();
        for (i, o) in &state.demos {
            self.demos.push((i.to_string(), o.to_string()));
        }

        true
    }
}

impl<'a, L, V> ComposableModule for PredictModule<'a, L, V>
where
    L: crate::recursive::Llm + 'static,
    V: crate::recursive::Validate + 'static,
{
    type Input<'b>
        = &'b str
    where
        Self: 'b;
    type Output<'b>
        = String
    where
        Self: 'b;
    type ForwardFut<'b>
        = std::pin::Pin<Box<dyn Future<Output = crate::error::Result<String>> + Send + 'b>>
    where
        Self: 'b;

    fn forward<'b>(&'b self, input: Self::Input<'b>) -> Self::ForwardFut<'b> {
        let prompt = self.build_prompt(input);
        Box::pin(async move {
            let output = self.llm.generate(&prompt, "", None).await?;
            Ok(output.text.to_string())
        })
    }

    fn name(&self) -> &str {
        self.name
    }

    fn save_state(&self) -> ModuleState<'_> {
        self.save_composable_state()
    }

    fn load_state(&mut self, state: &ModuleState<'_>) -> bool {
        self.load_composable_state(state)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // ModuleState tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_module_state_new() {
        let state = ModuleState::new("test");
        assert_eq!(state.name, "test");
        assert!(state.instructions.is_none());
        assert!(state.demos.is_empty());
        assert!(state.children.is_empty());
    }

    #[test]
    fn test_module_state_builder() {
        let state = ModuleState::new("predict")
            .with_instructions("Translate to French.")
            .with_demo("Hello", "Bonjour")
            .with_demo("Goodbye", "Au revoir")
            .with_child(ModuleState::new("retrieve"));

        assert_eq!(state.instructions.as_deref(), Some("Translate to French."));
        assert_eq!(state.demos.len(), 2);
        assert_eq!(state.demos[0].0, "Hello");
        assert_eq!(state.demos[0].1, "Bonjour");
        assert_eq!(state.demos[1].0, "Goodbye");
        assert_eq!(state.demos[1].1, "Au revoir");
        assert_eq!(state.children.len(), 1);
        assert_eq!(state.children[0].name, "retrieve");
    }

    #[test]
    fn test_module_state_into_owned() {
        let state = ModuleState::new("test")
            .with_instructions("instr")
            .with_demo("in", "out")
            .with_child(ModuleState::new("child").with_instructions("child_instr"));

        let owned: ModuleState<'static> = state.into_owned();

        assert_eq!(owned.name, "test");
        assert_eq!(owned.instructions.as_deref(), Some("instr"));
        assert_eq!(owned.demos.len(), 1);
        assert_eq!(owned.children.len(), 1);
        assert_eq!(
            owned.children[0].instructions.as_deref(),
            Some("child_instr")
        );
    }

    #[test]
    fn test_module_state_child_lookup() {
        let state = ModuleState::new("root")
            .with_child(ModuleState::new("a").with_instructions("inst_a"))
            .with_child(ModuleState::new("b"));

        assert!(state.child("a").is_some());
        assert_eq!(
            state.child("a").unwrap().instructions.as_deref(),
            Some("inst_a")
        );
        assert!(state.child("b").is_some());
        assert!(state.child("c").is_none());
    }

    #[test]
    fn test_module_state_child_mut() {
        let mut state = ModuleState::new("root").with_child(ModuleState::new("a"));

        let child = state.child_mut("a").unwrap();
        child.instructions = Some(Cow::Owned("updated".to_string()));

        assert_eq!(
            state.child("a").unwrap().instructions.as_deref(),
            Some("updated")
        );
    }

    // -----------------------------------------------------------------------
    // ModuleState serialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_module_state_json_roundtrip() {
        let state = ModuleState::new("predict")
            .with_instructions("Translate carefully.")
            .with_demo("cat", "chat")
            .with_child(
                ModuleState::new("retrieve").with_instructions("Search for relevant documents."),
            );

        let json = state.to_json().unwrap();
        let loaded = ModuleState::from_json(&json).unwrap();

        assert_eq!(loaded.name, "predict");
        assert_eq!(loaded.instructions.as_deref(), Some("Translate carefully."));
        assert_eq!(loaded.demos.len(), 1);
        assert_eq!(loaded.demos[0].0, "cat");
        assert_eq!(loaded.demos[0].1, "chat");
        assert_eq!(loaded.children.len(), 1);
        assert_eq!(loaded.children[0].name, "retrieve");
    }

    #[test]
    fn test_module_state_json_minimal() {
        // Only name, no optional fields
        let state = ModuleState::new("bare");
        let json = state.to_json().unwrap();
        let loaded = ModuleState::from_json(&json).unwrap();

        assert_eq!(loaded.name, "bare");
        assert!(loaded.instructions.is_none());
        assert!(loaded.demos.is_empty());
        assert!(loaded.children.is_empty());
    }

    #[test]
    fn test_module_state_nested_json() {
        let state = ModuleState::new("root").with_child(
            ModuleState::new("l1").with_child(ModuleState::new("l2").with_instructions("deep")),
        );

        let json = state.to_json().unwrap();
        let loaded = ModuleState::from_json(&json).unwrap();

        assert_eq!(
            loaded.children[0].children[0].instructions.as_deref(),
            Some("deep")
        );
    }

    // -----------------------------------------------------------------------
    // kkachi_module! macro tests
    // -----------------------------------------------------------------------

    // Define a leaf module for testing
    struct LeafModule {
        name: &'static str,
        instruction: String,
    }

    impl LeafModule {
        fn new(name: &'static str, instruction: &str) -> Self {
            Self {
                name,
                instruction: instruction.to_string(),
            }
        }

        fn save_composable_state(&self) -> ModuleState<'_> {
            ModuleState::new(self.name).with_instructions(Cow::Borrowed(self.instruction.as_str()))
        }

        fn load_composable_state(&mut self, state: &ModuleState<'_>) -> bool {
            if state.name != self.name {
                return false;
            }
            if let Some(ref instr) = state.instructions {
                self.instruction = instr.to_string();
            }
            true
        }
    }

    crate::kkachi_module! {
        struct TestComposite {
            #[module] retriever: LeafModule,
            #[module] generator: LeafModule,
            k: usize,
        }
    }

    #[test]
    fn test_macro_save_state() {
        let composite = TestComposite {
            retriever: LeafModule::new("retriever", "Find docs"),
            generator: LeafModule::new("generator", "Generate answer"),
            k: 5,
        };

        let state = composite.save_composable_state();
        assert_eq!(state.name, "TestComposite");
        assert_eq!(state.children.len(), 2);
        assert_eq!(state.children[0].name, "retriever");
        assert_eq!(state.children[0].instructions.as_deref(), Some("Find docs"));
        assert_eq!(state.children[1].name, "generator");
        assert_eq!(
            state.children[1].instructions.as_deref(),
            Some("Generate answer")
        );
    }

    #[test]
    fn test_macro_load_state() {
        let mut composite = TestComposite {
            retriever: LeafModule::new("retriever", "old"),
            generator: LeafModule::new("generator", "old"),
            k: 5,
        };

        let state = ModuleState::new("TestComposite")
            .with_child(ModuleState::new("retriever").with_instructions("new retriever instr"))
            .with_child(ModuleState::new("generator").with_instructions("new generator instr"));

        let ok = composite.load_composable_state(&state);
        assert!(ok);
        assert_eq!(composite.retriever.instruction, "new retriever instr");
        assert_eq!(composite.generator.instruction, "new generator instr");
        // Non-module field should be unchanged
        assert_eq!(composite.k, 5);
    }

    #[test]
    fn test_macro_load_wrong_name() {
        let mut composite = TestComposite {
            retriever: LeafModule::new("retriever", "old"),
            generator: LeafModule::new("generator", "old"),
            k: 3,
        };

        let state = ModuleState::new("WrongName");
        let ok = composite.load_composable_state(&state);
        assert!(!ok);
        // State should be unchanged
        assert_eq!(composite.retriever.instruction, "old");
    }

    #[test]
    fn test_macro_save_load_roundtrip() {
        let composite = TestComposite {
            retriever: LeafModule::new("retriever", "Search the database"),
            generator: LeafModule::new("generator", "Synthesize an answer"),
            k: 10,
        };

        let state = composite.save_composable_state();
        let json = state.to_json().unwrap();
        let loaded_state = ModuleState::from_json(&json).unwrap();

        let mut new_composite = TestComposite {
            retriever: LeafModule::new("retriever", ""),
            generator: LeafModule::new("generator", ""),
            k: 10,
        };

        let ok = new_composite.load_composable_state(&loaded_state);
        assert!(ok);
        assert_eq!(new_composite.retriever.instruction, "Search the database");
        assert_eq!(new_composite.generator.instruction, "Synthesize an answer");
    }

    // -----------------------------------------------------------------------
    // PredictModule tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_predict_module_build_prompt() {
        let llm = crate::recursive::MockLlm::new(|_, _| "mock".to_string());
        let module = PredictModule::new("qa", &llm, "Answer the question.")
            .demo("What is 1+1?", "2")
            .demo("What color is the sky?", "Blue");

        let prompt = module.build_prompt("What is Rust?");
        assert!(prompt.contains("Answer the question."));
        assert!(prompt.contains("Input: What is 1+1?"));
        assert!(prompt.contains("Output: 2"));
        assert!(prompt.contains("Input: What color is the sky?"));
        assert!(prompt.contains("Output: Blue"));
        assert!(prompt.contains("Input: What is Rust?"));
        assert!(prompt.ends_with("Output:"));
    }

    #[test]
    fn test_predict_module_save_load_state() {
        let llm = crate::recursive::MockLlm::new(|_, _| "mock".to_string());
        let module = PredictModule::new("qa", &llm, "Original instruction").demo("in1", "out1");

        let state = module.save_composable_state();
        assert_eq!(state.name, "qa");
        assert_eq!(state.instructions.as_deref(), Some("Original instruction"));
        assert_eq!(state.demos.len(), 1);

        // Load new state
        let mut module2 = PredictModule::new("qa", &llm, "placeholder");
        let new_state = ModuleState::new("qa")
            .with_instructions("Updated instruction")
            .with_demo("new_in", "new_out");

        let ok = module2.load_composable_state(&new_state);
        assert!(ok);
        assert_eq!(module2.instruction, "Updated instruction");
        assert_eq!(module2.demos.len(), 1);
        assert_eq!(module2.demos[0].0, "new_in");
        assert_eq!(module2.demos[0].1, "new_out");
    }

    #[test]
    fn test_predict_module_load_wrong_name() {
        let llm = crate::recursive::MockLlm::new(|_, _| "mock".to_string());
        let mut module = PredictModule::new("qa", &llm, "original");

        let state = ModuleState::new("wrong_name").with_instructions("should not apply");

        let ok = module.load_composable_state(&state);
        assert!(!ok);
        assert_eq!(module.instruction, "original");
    }

    #[tokio::test]
    async fn test_predict_module_forward() {
        let llm = crate::recursive::MockLlm::new(|prompt, _| {
            if prompt.contains("Capital of France") {
                "Paris".to_string()
            } else {
                "Unknown".to_string()
            }
        });

        let module = PredictModule::new("geo", &llm, "Answer the geography question.");

        let result = ComposableModule::forward(&module, "Capital of France").await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Paris");
    }

    #[tokio::test]
    async fn test_predict_module_with_demos_forward() {
        let llm = crate::recursive::MockLlm::new(|prompt, _| {
            // Verify demos are in the prompt
            if prompt.contains("Input: dog") && prompt.contains("Output: chien") {
                "chat".to_string()
            } else {
                "no demos found".to_string()
            }
        });

        let module = PredictModule::new("translate", &llm, "Translate English to French.")
            .demo("dog", "chien");

        let result = ComposableModule::forward(&module, "cat").await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "chat");
    }

    #[test]
    fn test_predict_module_composable_trait_name() {
        let llm = crate::recursive::MockLlm::new(|_, _| "x".to_string());
        let module = PredictModule::new("my_predictor", &llm, "instr");
        assert_eq!(ComposableModule::name(&module), "my_predictor");
    }
}
