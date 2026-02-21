// Mock data constants
export const MOCK_ALL_MODELS = [
  {
    model_id: "GradientBoosting_b22de4e9f16e_1771603320",
    model_name: "GradientBoosting",
    task: "classification",
    cv_score_mean: 0.9963594852635949,
    cv_score_std: 0.0034026260909819847,
    test_score: 1,
    latency_ms: 0.006632804870605469,
    size_mb: 3,
    final_score: 0.8499993367195129,
    hyperparameters: {
      n_estimators: 106,
      learning_rate: 0.2835559696877989,
      max_depth: 10,
      min_samples_split: 3,
      subsample: 0.6014947911252734
    },
    dataset_hash: "b22de4e9f16e",
    timestamp: "2026-02-20T21:32:00.990119",
    feature_count: 136,
    train_samples: 1097,
    pickle_download: "/models/GradientBoosting_b22de4e9f16e_1771603320/download"
  },
  {
    model_id: "GradientBoosting_b22de4e9f16e_1771615752",
    model_name: "GradientBoosting",
    task: "classification",
    cv_score_mean: 0.9972685761726858,
    cv_score_std: 0.0022302110935331866,
    test_score: 0.9963636363636363,
    latency_ms: 0.0053310394287109375,
    size_mb: 3,
    final_score: 0.8463631032596934,
    hyperparameters: {
      n_estimators: 195,
      learning_rate: 0.07300813208006586,
      max_depth: 5,
      min_samples_split: 5,
      subsample: 0.7288875696068378
    },
    dataset_hash: "b22de4e9f16e",
    timestamp: "2026-02-21T00:59:12.944102",
    feature_count: 136,
    train_samples: 1097,
    pickle_download: "/models/GradientBoosting_b22de4e9f16e_1771615752/download"
  }
];

// Shared glass card styles
export const GLASS_CARD_STYLE = {
  background: 'linear-gradient(135deg, rgba(117, 117, 117, 0.2) 0%, rgba(82, 82, 82, 0.15) 100%)',
  backdropFilter: 'blur(20px)',
  border: '1px solid rgba(140, 140, 140, 0.3)',
  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)'
};
