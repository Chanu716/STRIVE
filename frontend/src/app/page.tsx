"use client";

import { motion } from "framer-motion";
import { useRouter } from "next/navigation";
import { ShieldCheck, MapPin, Activity, ArrowRight, ChevronRight } from "lucide-react";
import { BorderGlow } from "@/components/ui/BorderGlow";
import LiquidEther from "@/components/LiquidEther";

const fadeUp = {
  hidden: { opacity: 0, y: 30 },
  show: { opacity: 1, y: 0, transition: { duration: 0.6 } }
};

const staggerContainer = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.15 } }
};

export default function LandingPage() {
  const router = useRouter();

  return (
    <div style={{ minHeight: "100vh", position: "relative", zIndex: 0, overflow: "hidden", display: "flex", flexDirection: "column" }}>
      {/* LiquidEther React-Bits Absolute Background */}
      <div style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", zIndex: -1 }}>
        <LiquidEther
          colors={['#722F37', '#FFFFFF', '#Fdf8f9']}
          mouseForce={25}
          cursorSize={120}
          isViscous={true}
          viscous={30}
          iterationsViscous={32}
          iterationsPoisson={32}
          resolution={0.5}
          isBounce={false}
          autoDemo={true}
          autoSpeed={0.4}
          autoIntensity={1.8}
          takeoverDuration={0.4}
          autoResumeDelay={2500}
          autoRampDuration={1.0}
        />
      </div>

      {/* Navigation */}
      <header style={{ padding: "1.5rem 3rem", display: "flex", justifyContent: "space-between", alignItems: "center", zIndex: 10 }}>
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <img src="/STRIVE.png" alt="STRIVE Logo" style={{ height: "45px", objectFit: "contain" }} />
          <span style={{ fontWeight: 900, fontSize: "1.5rem", color: "var(--wine-primary)", letterSpacing: "-0.04em" }}>STRIVE</span>
        </div>
        <div>
          <button
            onClick={() => router.push("/login")}
            className="text-meta card-glassy"
            style={{
              color: "var(--text-primary)", fontWeight: 800,
              padding: "0.75rem 1.75rem", display: "flex", alignItems: "center", gap: "8px"
            }}
          >
            LOGIN <ChevronRight size={16} />
          </button>
        </div>
      </header>

      {/* Hero Section */}
      <main style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "4rem 2rem", textAlign: "center", zIndex: 10 }}>
        <motion.div variants={staggerContainer} initial="hidden" animate="show" style={{ maxWidth: "800px" }}>

          <motion.div variants={fadeUp} style={{ marginBottom: "1.5rem" }}>
            <span style={{ display: "inline-flex", alignItems: "center", gap: "8px", padding: "0.5rem 1rem", background: "var(--wine-subtle)", color: "var(--wine-primary)", borderRadius: "2rem", fontSize: "0.85rem", fontWeight: 700, letterSpacing: "0.05em" }}>
              <span style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--wine-primary)" }} />
              ENTERPRISE GEOMATICS
            </span>
          </motion.div>

          <motion.h1 variants={fadeUp} style={{ fontSize: "5rem", fontWeight: 900, letterSpacing: "-0.04em", lineHeight: 1.1, color: "var(--text-primary)", marginBottom: "1.5rem" }}>
            Intelligent Safety Routing for the Modern Network.
          </motion.h1>

          <motion.p variants={fadeUp} style={{ fontSize: "1.25rem", color: "var(--text-secondary)", fontWeight: 500, lineHeight: 1.6, marginBottom: "3rem", padding: "0 2rem" }}>
            STRIVE leverages real-world NHTSA FARS validation and gradient-boosted diagnostic explainability (SHAP) to assess dynamic route risk with surgical precision.
          </motion.p>

          <motion.div variants={fadeUp}>
            <button
              onClick={() => router.push("/login")}
              style={{
                background: "var(--wine-primary)", color: "#fff", border: "none",
                padding: "1.25rem 2.5rem", borderRadius: "1rem", fontSize: "1.1rem", fontWeight: 800,
                cursor: "pointer", display: "inline-flex", alignItems: "center", gap: "12px",
                boxShadow: "0 10px 25px var(--wine-glow)", transition: "transform 0.2s, box-shadow 0.2s"
              }}
              onMouseOver={(e) => { e.currentTarget.style.transform = "translateY(-2px)"; e.currentTarget.style.boxShadow = "0 15px 30px var(--wine-glow)"; }}
              onMouseOut={(e) => { e.currentTarget.style.transform = "none"; e.currentTarget.style.boxShadow = "0 10px 25px var(--wine-glow)"; }}
            >
              Access Command Center <ArrowRight size={20} />
            </button>
          </motion.div>

        </motion.div>

        {/* Feature Highlights */}
        <motion.div variants={staggerContainer} initial="hidden" animate="show" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "2rem", width: "100%", maxWidth: "1200px", marginTop: "6rem", textAlign: "left" }}>

          <motion.div variants={fadeUp}>
            <BorderGlow glowColor="var(--wine-glow)">
              <div className="card-glassy" style={{ padding: "2rem", height: "100%" }}>
                <div style={{ padding: "0.75rem", background: "rgba(114, 47, 55, 0.08)", display: "inline-block", borderRadius: "12px", marginBottom: "1.5rem" }}>
                  <ShieldCheck size={28} style={{ color: "var(--wine-primary)" }} />
                </div>
                <h3 style={{ fontSize: "1.2rem", fontWeight: 800, marginBottom: "0.5rem" }}>SHAP Diagnostic ML</h3>
                <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", lineHeight: 1.5 }}>
                  Every risk vector is entirely transparent. See precisely how weather, light, and historical accident densities contribute to XGBoost risk inferences.
                </p>
              </div>
            </BorderGlow>
          </motion.div>

          <motion.div variants={fadeUp}>
            <BorderGlow glowColor="rgba(5, 150, 105, 0.4)">
              <div className="card-glassy" style={{ padding: "2rem", height: "100%" }}>
                <div style={{ padding: "0.75rem", background: "rgba(5, 150, 105, 0.1)", display: "inline-block", borderRadius: "12px", marginBottom: "1.5rem" }}>
                  <MapPin size={28} style={{ color: "var(--accent-emerald)" }} />
                </div>
                <h3 style={{ fontSize: "1.2rem", fontWeight: 800, marginBottom: "0.5rem" }}>Real-World Data Engine</h3>
                <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", lineHeight: 1.5 }}>
                  Powered exclusively by genuine NHTSA FARS collision telemetry and parsed OpenStreetMap drive networks; entirely synthetic-free reliability.
                </p>
              </div>
            </BorderGlow>
          </motion.div>

          <motion.div variants={fadeUp}>
            <BorderGlow glowColor="rgba(14, 165, 233, 0.4)">
              <div className="card-glassy" style={{ padding: "2rem", height: "100%" }}>
                <div style={{ padding: "0.75rem", background: "rgba(14, 165, 233, 0.1)", display: "inline-block", borderRadius: "12px", marginBottom: "1.5rem" }}>
                  <Activity size={28} style={{ color: "var(--accent-blue)" }} />
                </div>
                <h3 style={{ fontSize: "1.2rem", fontWeight: 800, marginBottom: "0.5rem" }}>Live Geospatial Canvas</h3>
                <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", lineHeight: 1.5 }}>
                  Hardware-accelerated MapLibre views integrate active driver coordinates directly against scored segment metadata.
                </p>
              </div>
            </BorderGlow>
          </motion.div>

        </motion.div>
      </main>

      {/* Footer */}
      <footer style={{ padding: "2rem", textAlign: "center", borderTop: "1px solid var(--border-subtle)", color: "var(--text-secondary)", fontSize: "0.85rem", fontWeight: 500 }}>
        © {new Date().getFullYear()} STRIVE AI Traffic Core. All rights reserved. Highly confidential routing architecture.
      </footer>
    </div>
  );
}
