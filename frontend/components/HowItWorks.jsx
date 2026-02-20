import { Upload, Settings, Brain, Rocket, ArrowRight } from 'lucide-react';

export default function HowItWorks() {
  const steps = [
    { step: "01", title: "Upload CSV", desc: "Drag & drop your dataset. Any format, any size up to 10GB.", Icon: Upload },
    { step: "02", title: "Configure", desc: "Set target column, task type, and optimization metric.", Icon: Settings },
    { step: "03", title: "AutoTrain", desc: "Watch the pipeline run: preprocessing, tuning, selection.", Icon: Brain },
    { step: "04", title: "Deploy", desc: "Export model + get a live REST API endpoint instantly.", Icon: Rocket },
  ];

  return (
    <section className="relative z-10 max-w-7xl mx-auto px-8 py-24">
      <div className="text-center mb-16">
        <h2 
          className="font-bold mb-4" 
          style={{ fontSize: "clamp(28px,4vw,40px)", color: "#E0E0E0", letterSpacing: "-0.01em" }}
        >
          Zero to Production in 4 Steps
        </h2>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {steps.map(({ step, title, desc, Icon }, i) => (
          <div key={step} className="relative">
            {i < 3 && (
              <div className="hidden md:flex absolute top-1/2 -translate-y-1/2 left-full w-6 h-full items-center justify-center z-10">
                <ArrowRight className="text-[#757575]" size={20} />
              </div>
            )}
            <div className="glass-card glass-card-hover rounded-[18px] p-6 relative overflow-hidden">
              <div 
                className="absolute top-0 left-0 right-0 h-px" 
                style={{ background: "linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent)" }} 
              />
              <div className="text-xs font-bold mb-3" style={{ color: "#757575" }}>{step}</div>
              <div className="mb-3">
                <Icon className="text-[#B0B0B0]" size={28} />
              </div>
              <h3 className="font-semibold mb-2" style={{ color: "#E0E0E0" }}>{title}</h3>
              <p className="text-sm" style={{ color: "#8C8C8C" }}>{desc}</p>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
