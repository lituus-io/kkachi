// Copyright © 2025 lituus-io <spicyzhug@gmail.com>
// All Rights Reserved.
// Licensed under PolyForm Noncommercial 1.0.0

//! Model configuration commands.

use super::{Command, ExecutionContext, Output};
use crate::repl::{ProviderType, SessionState};
use console::style;

/// Model command - set the model name.
pub struct ModelCommand;

impl Command for ModelCommand {
    fn name(&self) -> &str {
        "model"
    }

    fn aliases(&self) -> &[&str] {
        &["m"]
    }

    fn description(&self) -> &str {
        "Set the LLM model"
    }

    fn help(&self) -> &str {
        "Usage: model <name>\n\n\
         Set the model to use for predictions.\n\n\
         Examples:\n\
           model gpt-4o\n\
           model anthropic-sonnet\n\
           model gemini-pro\n\
           model llama3:8b"
    }

    fn completions(&self, partial: &str, state: &SessionState) -> Vec<String> {
        let models = match state.provider {
            ProviderType::OpenAI => vec!["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
            ProviderType::Anthropic => {
                vec!["anthropic-sonnet", "anthropic-opus", "anthropic-haiku"]
            }
            ProviderType::Google => vec!["gemini-pro", "gemini-pro-vision", "gemini-1.5-pro"],
            ProviderType::Local => vec!["llama3:8b", "llama3:70b", "mistral:7b", "codellama:7b"],
        };

        models
            .into_iter()
            .filter(|m| m.starts_with(partial))
            .map(|s| s.to_string())
            .collect()
    }

    fn execute(
        &self,
        args: &str,
        state: &mut SessionState,
        _ctx: &mut ExecutionContext<'_>,
    ) -> Output {
        let args = args.trim();

        if args.is_empty() {
            return Output::text(format!(
                "Current model: {} ({})",
                state.lm_config.model,
                state.provider.display_name()
            ));
        }

        state.lm_config.model = args.to_string();
        Output::success(format!("Model set: {}", args))
    }
}

/// Provider command - set the LLM provider.
pub struct ProviderCommand;

impl Command for ProviderCommand {
    fn name(&self) -> &str {
        "provider"
    }

    fn aliases(&self) -> &[&str] {
        &["p"]
    }

    fn description(&self) -> &str {
        "Set the LLM provider"
    }

    fn help(&self) -> &str {
        "Usage: provider <name>\n\n\
         Set the provider for LLM calls.\n\n\
         Available providers:\n\
           openai     - OpenAI (GPT models)\n\
           anthropic  - Anthropic\n\
           google     - Google (Gemini models)\n\
           local      - Local models (Ollama, vLLM, etc.)"
    }

    fn completions(&self, partial: &str, _state: &SessionState) -> Vec<String> {
        vec!["openai", "anthropic", "google", "local"]
            .into_iter()
            .filter(|p| p.starts_with(partial))
            .map(|s| s.to_string())
            .collect()
    }

    fn execute(
        &self,
        args: &str,
        state: &mut SessionState,
        _ctx: &mut ExecutionContext<'_>,
    ) -> Output {
        let args = args.trim();

        if args.is_empty() {
            return Output::text(format!(
                "Current provider: {}",
                state.provider.display_name()
            ));
        }

        if let Some(provider) = ProviderType::from_str(args) {
            state.provider = provider;
            Output::success(format!("Provider set: {}", provider.display_name()))
        } else {
            Output::error("Unknown provider. Use: openai, anthropic, google, or local")
        }
    }
}

/// Temperature command - set the temperature.
pub struct TemperatureCommand;

impl Command for TemperatureCommand {
    fn name(&self) -> &str {
        "temperature"
    }

    fn aliases(&self) -> &[&str] {
        &["temp", "t"]
    }

    fn description(&self) -> &str {
        "Set the temperature"
    }

    fn help(&self) -> &str {
        "Usage: temperature <value>\n\n\
         Set the temperature for LLM generation (0.0 - 2.0).\n\
         Lower = more deterministic, higher = more creative.\n\n\
         Examples:\n\
           temperature 0.7   (default, balanced)\n\
           temperature 0.0   (deterministic)\n\
           temperature 1.5   (creative)"
    }

    fn execute(
        &self,
        args: &str,
        state: &mut SessionState,
        _ctx: &mut ExecutionContext<'_>,
    ) -> Output {
        let args = args.trim();

        if args.is_empty() {
            return Output::text(format!(
                "Current temperature: {}",
                state.lm_config.temperature
            ));
        }

        let temp: f32 = match args.parse() {
            Ok(t) => t,
            Err(_) => return Output::error("Invalid temperature. Use a number like 0.7"),
        };

        if !(0.0..=2.0).contains(&temp) {
            return Output::warning(format!(
                "Temperature {} is outside typical range (0.0 - 2.0)",
                temp
            ));
        }

        state.lm_config.temperature = temp;
        Output::success(format!("Temperature set: {}", temp))
    }
}

/// Config command - show full configuration.
pub struct ConfigCommand;

impl Command for ConfigCommand {
    fn name(&self) -> &str {
        "config"
    }

    fn aliases(&self) -> &[&str] {
        &["cfg"]
    }

    fn description(&self) -> &str {
        "Show full configuration"
    }

    fn execute(
        &self,
        _args: &str,
        state: &mut SessionState,
        _ctx: &mut ExecutionContext<'_>,
    ) -> Output {
        let mut output = String::new();

        output.push_str(&format!(
            "{}\n\n",
            style("LLM CONFIGURATION").bold().underlined()
        ));

        output.push_str(&format!(
            "  {} {}\n",
            style("Provider:").bold(),
            state.provider.display_name()
        ));

        output.push_str(&format!(
            "  {} {}\n",
            style("Model:").bold(),
            state.lm_config.model
        ));

        output.push_str(&format!(
            "  {} {}\n",
            style("Temperature:").bold(),
            state.lm_config.temperature
        ));

        if let Some(max_tokens) = state.lm_config.max_tokens {
            output.push_str(&format!(
                "  {} {}\n",
                style("Max Tokens:").bold(),
                max_tokens
            ));
        }

        if let Some(top_p) = state.lm_config.top_p {
            output.push_str(&format!("  {} {}\n", style("Top P:").bold(), top_p));
        }

        if let Some(ref base_url) = state.lm_config.base_url {
            output.push_str(&format!("  {} {}\n", style("Base URL:").bold(), base_url));
        }

        output.push_str(&format!(
            "  {} {}\n",
            style("API Key:").bold(),
            if state.lm_config.api_key.is_some() {
                "set (custom)"
            } else {
                "from environment"
            }
        ));

        Output::text(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repl::StateHistory;
    use kkachi::DiffRenderer;

    fn make_ctx<'a>(
        history: &'a mut StateHistory,
        renderer: &'a DiffRenderer,
    ) -> ExecutionContext<'a> {
        ExecutionContext { renderer, history }
    }

    #[test]
    fn test_model_command() {
        let cmd = ModelCommand;
        let mut state = SessionState::default();
        let renderer = DiffRenderer::new();
        let mut history = StateHistory::new();
        let mut ctx = make_ctx(&mut history, &renderer);

        let output = cmd.execute("gpt-4o", &mut state, &mut ctx);
        assert!(matches!(output, Output::Success(_)));
        assert_eq!(state.lm_config.model, "gpt-4o");
    }

    #[test]
    fn test_provider_command() {
        let cmd = ProviderCommand;
        let mut state = SessionState::default();
        let renderer = DiffRenderer::new();
        let mut history = StateHistory::new();
        let mut ctx = make_ctx(&mut history, &renderer);

        let output = cmd.execute("anthropic", &mut state, &mut ctx);
        assert!(matches!(output, Output::Success(_)));
        assert_eq!(state.provider, ProviderType::Anthropic);
    }

    #[test]
    fn test_temperature_command() {
        let cmd = TemperatureCommand;
        let mut state = SessionState::default();
        let renderer = DiffRenderer::new();
        let mut history = StateHistory::new();
        let mut ctx = make_ctx(&mut history, &renderer);

        let output = cmd.execute("0.5", &mut state, &mut ctx);
        assert!(matches!(output, Output::Success(_)));
        assert_eq!(state.lm_config.temperature, 0.5);
    }

    #[test]
    fn test_config_command() {
        let cmd = ConfigCommand;
        let mut state = SessionState::default();
        let renderer = DiffRenderer::new();
        let mut history = StateHistory::new();
        let mut ctx = make_ctx(&mut history, &renderer);

        let output = cmd.execute("", &mut state, &mut ctx);
        assert!(matches!(output, Output::Text(_)));
    }
}
