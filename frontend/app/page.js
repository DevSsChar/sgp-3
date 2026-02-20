"use client";
import { useRef } from "react";
import Navbar from "@/components/Navbar";
import Hero from "@/components/Hero";
import Features from "@/components/Features";
import HowItWorks from "@/components/HowItWorks";
import CTA from "@/components/CTA";
import Footer from "@/components/Footer";
import ParticleBackground from "@/components/ui/ParticleBackground";

export default function LandingPage() {
  const particleRef = useRef(null);

  const handleMouseMove = (e) => {
    if (particleRef.current) {
      particleRef.current.updateMouse(e.clientX, e.clientY);
    }
  };

  const handleMouseLeave = () => {
    if (particleRef.current) {
      particleRef.current.deactivateMouse();
    }
  };

  return (
    <div 
      className="min-h-screen relative overflow-hidden"
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
    >
      {/* Interactive Particle Background */}
      <ParticleBackground ref={particleRef} />
      
      {/* Background grid overlay */}
      <div className="fixed inset-0 bg-grid-pattern pointer-events-none z-[1]" />

      {/* Content wrapper with higher z-index */}
      <div className="relative z-10">
        <Navbar />
        <Hero />
        <Features />
        <HowItWorks />
        <CTA />
        <Footer />
      </div>
    </div>
  );
}

