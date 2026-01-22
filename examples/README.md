# Kkachi Examples - Automated Prompt Optimization Pipelines

This directory contains comprehensive examples demonstrating automated prompt optimization pipelines based on user feedback and production data.

## Examples Overview

### 1. Continuous Learning Pipeline
**File**: `continuous_learning.rs`

A production-ready example showing real-time feedback collection and automatic model retraining.

**Features**:
- Real-time feedback collection
- Automatic retraining when accuracy drops
- Thread-safe feedback storage
- Zero-downtime model updates
- Email spam classification use case

**Run**:
```bash
cd examples
cargo run --example continuous_learning
```

**Key Concepts**:
- `FeedbackStore`: Thread-safe storage for user corrections
- `ContinuousLearner`: Orchestrates feedback collection and retraining
- Automatic trigger when correction threshold is reached
- Model hot-swapping without downtime

### 2. Automated Optimization Pipeline
**File**: `automated_optimization_pipeline.rs`

Complete end-to-end optimization pipeline with multiple feedback cycles.

**Features**:
- Multi-stage feedback collection
- Iterative optimization cycles
- Performance tracking across iterations
- Sentiment analysis use case

**Run**:
```bash
cd examples
cargo run --example automated_optimization_pipeline
```

**Pipeline Stages**:
1. Initial deployment and feedback collection
2. First optimization cycle
3. Deploy optimized model
4. Collect more feedback
5. Second optimization cycle
6. Final evaluation
7. Production deployment

**Key Concepts**:
- `OptimizationPipeline`: Complete feedback-to-deployment flow
- `FeedbackCollector`: Tracks predictions and corrections
- `OptimizationRun`: Metrics for each optimization cycle
- Custom metrics (SentimentAccuracy)

### 3. Production Pipeline with A/B Testing
**File**: `production_pipeline.rs`

Enterprise-grade pipeline with A/B testing, performance monitoring, and automated rollback.

**Features**:
- A/B testing between models
- Performance monitoring (accuracy + latency)
- Automated rollback on regression
- Comprehensive evaluation
- Question answering use case

**Run**:
```bash
cd examples
cargo run --example production_pipeline
```

**Pipeline Components**:
1. **ModelVersion**: Tracks model name, accuracy, and latency
2. **ProductionPipeline**: Manages active and candidate models
3. **A/B Testing**: Statistical comparison before promotion
4. **Rollback Logic**: Protects against accuracy or latency regression

**Decision Criteria**:
- Promote if accuracy improves > 5% and latency doesn't regress > 50ms
- Reject if accuracy drops > 5%
- Reject if latency increases > 100ms
- Otherwise keep current model

## Common Patterns

### 1. Feedback Collection

All examples demonstrate collecting user feedback in production:

```rust
struct FeedbackCollector {
    feedback: Vec<UserFeedback>,
}

impl FeedbackCollector {
    fn add_feedback(&mut self, input: String, predicted: String, correct: String) {
        // Store correction
    }

    fn build_training_set(&self) -> Vec<Example<'static>> {
        // Convert feedback to training examples
    }
}
```

### 2. Model Optimization

Using BootstrapFewShot optimizer:

```rust
let config = OptimizerConfig {
    max_iterations: 2,
    batch_size: training_set.len(),
    seed: 42,
    metric_threshold: Some(0.8),
};

let optimizer = BootstrapFewShot::new(config)
    .with_max_demos(5);

let optimized = optimizer.optimize(baseline, &training_set).await?;
```

### 3. Evaluation and Metrics

Custom metrics for domain-specific evaluation:

```rust
struct SentimentAccuracy;

impl Metric for SentimentAccuracy {
    fn evaluate<'a>(&self, example: &Example<'a>, prediction: &Prediction<'a>) -> MetricResult {
        let expected = example.get_output("sentiment");
        let actual = prediction.get("sentiment");

        let passed = match (expected, actual) {
            (Some(exp), Some(act)) => exp.trim().eq_ignore_ascii_case(act.trim()),
            _ => false,
        };

        MetricResult {
            score: if passed { 1.0 } else { 0.0 },
            passed,
            details: Some(format!("Expected: {:?}, Got: {:?}", expected, actual)),
        }
    }

    fn name(&self) -> &str {
        "SentimentAccuracy"
    }
}
```

### 4. Production Deployment

Thread-safe model swapping:

```rust
async fn retrain(&mut self) -> kkachi::Result<()> {
    let corrections = self.feedback_store.get_corrections();
    let training_set = self.build_training_set(corrections);

    let optimizer = BootstrapFewShot::new(config);
    let optimized = optimizer.optimize(baseline, &training_set).await?;

    // Hot-swap model
    self.current_model = optimized;

    Ok(())
}
```

## Architecture Patterns

### Continuous Learning Loop

```
┌─────────────────────────────────────────────────┐
│                                                 │
│  Production                                     │
│  ┌──────────┐                                   │
│  │  Model   │ ──► Predictions                   │
│  └──────────┘                                   │
│       ▲                                         │
│       │                                         │
│       │ Update                                  │
│       │                                         │
│  ┌──────────┐    ┌──────────────┐              │
│  │Optimizer │◄───│   Feedback   │◄── User      │
│  └──────────┘    │  Collector   │    Corrections│
│                  └──────────────┘              │
│                                                 │
└─────────────────────────────────────────────────┘
```

### A/B Testing Pipeline

```
                  ┌──────────────┐
                  │  Model A     │
                  │  (Active)    │
                  └──────────────┘
                         │
                         ├─── Test Set ───► Metrics A
                         │
                         ▼
                  ┌──────────────┐
                  │  Model B     │
                  │ (Candidate)  │
                  └──────────────┘
                         │
                         ├─── Test Set ───► Metrics B
                         │
                         ▼
                  ┌──────────────┐
                  │   Compare    │
                  │   Metrics    │
                  └──────────────┘
                         │
                    ┌────┴────┐
                    │         │
              Promote?    Rollback?
                    │         │
                    ▼         ▼
              Deploy B    Keep A
```

## Performance Considerations

### Zero-Copy Architecture

All examples leverage Kkachi's zero-copy design:

```rust
// Borrowed data - no allocation
let sig = Signature::parse("text -> sentiment")?;

// Owned when needed
let owned_sig = sig.into_owned();
```

### Async I/O

All LM calls are async, non-blocking:

```rust
let prediction = model.forward(inputs).await?;
```

### Parallel Evaluation

Use `ParallelEvaluator` for large datasets:

```rust
let evaluator = ParallelEvaluator::new(Arc::new(ExactMatch))
    .with_threads(8);

let results = evaluator.evaluate_predictions(&examples, &predictions)?;
```

## Production Deployment Checklist

- [x] **Feedback Collection**: Capture user corrections
- [x] **Monitoring**: Track accuracy and latency
- [x] **Automated Retraining**: Trigger on performance drop
- [x] **A/B Testing**: Compare before deploying
- [x] **Rollback**: Automated regression protection
- [x] **Thread Safety**: Concurrent access to models
- [x] **Zero Downtime**: Hot model swapping

## Integration with Real LM Providers

Replace `MockLM` with real providers:

```rust
use kkachi_client::{OpenAIProvider, LMConfig};

// OpenAI
let provider = OpenAIProvider::new(api_key, "https://api.openai.com/v1");
let lm = Arc::new(provider);

// Use in pipeline
let model = Predict::new(signature).with_lm(lm);
```

## Metrics and Monitoring

Track key metrics:

1. **Accuracy**: Percentage of correct predictions
2. **Latency**: Average response time
3. **Feedback Rate**: Corrections per prediction
4. **Improvement**: Accuracy gain after optimization

Example monitoring:

```rust
struct ModelMetrics {
    accuracy: f64,
    avg_latency_ms: f64,
    total_predictions: usize,
    total_corrections: usize,
}

impl ModelMetrics {
    fn feedback_rate(&self) -> f64 {
        self.total_corrections as f64 / self.total_predictions as f64
    }
}
```

## Advanced Use Cases

### 1. Multi-Model Ensemble

Combine predictions from multiple optimized models:

```rust
let models = vec![model_a, model_b, model_c];
let predictions: Vec<_> = models.iter()
    .map(|m| m.forward(inputs.clone()))
    .collect();

// Aggregate predictions
let ensemble_result = aggregate(predictions);
```

### 2. Domain-Specific Optimization

Optimize different models for different domains:

```rust
let spam_model = optimize_for("spam_detection", spam_examples);
let sentiment_model = optimize_for("sentiment", sentiment_examples);

// Route based on task
match task_type {
    "spam" => spam_model.forward(inputs).await,
    "sentiment" => sentiment_model.forward(inputs).await,
    _ => default_model.forward(inputs).await,
}
```

### 3. Continuous Improvement

Schedule periodic retraining:

```rust
// Every day at midnight
tokio::spawn(async move {
    loop {
        tokio::time::sleep(Duration::from_secs(86400)).await;
        pipeline.retrain_if_needed().await;
    }
});
```

## Testing

All examples include inline testing. To run:

```bash
# Run specific example
cargo run --example continuous_learning

# Build all examples
cargo build --examples

# Check for compilation errors
cargo check --examples
```

## Learn More

- **QUICK_START.md**: Basic Kkachi usage
- **ARCHITECTURE.md**: Deep dive into Kkachi design
- **TEST_SUMMARY.md**: Test coverage details
- **COMPLETION_REPORT.md**: Full implementation details

## License

MIT OR Apache-2.0
