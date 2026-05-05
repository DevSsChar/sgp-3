"use client";
import OptunaVisual from "@/components/OptunaVisual";
import DriftVisual from "@/components/DriftVisual";
import SmoteVisual from "@/components/SmoteVisual";

const FEATURES = [
  {
    icon: (
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
      </svg>
    ),
    title: "AutoML Pipeline",
    desc: "End-to-end automation from raw data ingestion to production model. Zero manual feature engineering.",
    visual: "pipeline",
  },
  {
    icon: (
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <circle cx="12" cy="12" r="3"/><path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83"/>
      </svg>
    ),
    title: "Optuna Hyperparameter Tuning",
    desc: "Bayesian optimization with 500+ trials. Automated search spaces. Best trial convergence guaranteed.",
    visual: "optuna",
  },
  {
    icon: (
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
      </svg>
    ),
    title: "Drift Detection",
    desc: "Real-time PSI monitoring. Statistical distribution tests. Automatic retraining triggers.",
    visual: "drift",
  },
  {
    icon: (
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75M21 21v-2a4 4 0 0 0-4-4H7a4 4 0 0 0-4 4v2"/>
      </svg>
    ),
    title: "SMOTE Balancing",
    desc: "Synthetic minority oversampling. Borderline-SMOTE variants. Automatic class imbalance correction.",
    visual: "smote",
  },
  {
    icon: (
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="9" y1="21" x2="9" y2="9"/>
      </svg>
    ),
    title: "Model Leaderboard",
    desc: "Side-by-side comparison across Accuracy, F1, Recall, AUC. One-click champion selection.",
    visual: "leaderboard",
  },
  {
    icon: (
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>
      </svg>
    ),
    title: "Cross-Validation Engine",
    desc: "Stratified K-Fold. Time-series splits. Nested CV for unbiased generalization estimates.",
    visual: null,
  },
  {
    icon: (
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
      </svg>
    ),
    title: "Explainability (SHAP)",
    desc: "SHAP values, feature importance plots, and decision boundary visualizations baked in.",
    visual: null,
  },
  {
    icon: (
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M18 20V10M12 20V4M6 20v-6"/>
      </svg>
    ),
    title: "Experiment Tracking",
    desc: "MLflow-compatible run history. Parameter snapshots. Model versioning with rollback.",
    visual: null,
  },
  {
    icon: (
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
        <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
      </svg>
    ),
    title: "One-Click Deployment",
    desc: "Export as .pkl or .joblib. Containerized REST API. Kubernetes-ready with autoscaling.",
    visual: null,
  },
];

export default function Features() {
  return (
    <section id="features" className="relative z-10 max-w-7xl mx-auto px-8 py-24">
      <div className="text-center mb-16">
        <div 
          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full mb-4 text-xs font-medium"
          style={{ background: "rgba(117,117,117,0.2)", border: "1px solid rgba(140,140,140,0.3)", color: "#B0B0B0" }}
        >
          Full Capability Suite
        </div>
        <h2 
          className="font-bold mb-4" 
          style={{ fontSize: "clamp(28px,4vw,40px)", color: "#E0E0E0", letterSpacing: "-0.01em" }}
        >
          Every Tool an ML Engineer Needs
        </h2>
        <p className="text-base max-w-2xl mx-auto" style={{ color: "#8C8C8C" }}>
          Built for production scale. Every component is optimized for enterprise ML teams running hundreds of experiments.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
        {FEATURES.map(({ icon, title, desc, visual }) => (
          <div
            key={title}
            className="glass-card glass-card-hover rounded-[18px] p-6 relative overflow-hidden group"
          >
            {/* Shine */}
            <div 
              className="absolute top-0 left-0 right-0 h-px" 
              style={{ background: "linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent)" }} 
            />
            <div
              className="w-10 h-10 rounded-xl flex items-center justify-center mb-4 transition-all duration-250"
              style={{ background: "rgba(82,82,82,0.6)", color: "#B0B0B0", border: "1px solid rgba(140,140,140,0.3)" }}
            >
              {icon}
            </div>
            <h3 className="font-semibold mb-2" style={{ fontSize: "15px", color: "#E0E0E0" }}>{title}</h3>
            <p className="text-sm leading-relaxed" style={{ color: "#8C8C8C" }}>{desc}</p>
            {visual === "optuna" && <OptunaVisual />}
            {visual === "drift" && <DriftVisual />}
            {visual === "smote" && <SmoteVisual />}
          </div>
        ))}
      </div>
    </section>
  );
}
