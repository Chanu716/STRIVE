"use client";

import { motion, AnimatePresence } from "framer-motion";
import { FileText, Info, X } from "lucide-react";

interface ShapModalProps {
  isOpen: boolean;
  onClose: () => void;
  data: {
    title: string;
    risk: string;
    label: string;
    metric: string;
    factors: Array<{ label: string; shap: number }>;
    summary: string;
  } | null;
}

export function ShapModal({ isOpen, onClose, data }: ShapModalProps) {
  if (!data) return null;

  const factors = data.factors || [];
  const summary = data.summary || "No detailed analysis available for this segment.";

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            style={{
              position: "fixed",
              top: 0, left: 0, right: 0, bottom: 0,
              backgroundColor: "rgba(0,0,0,0.5)",
              backdropFilter: "blur(4px)",
              zIndex: 9998
            }}
          />
          <motion.div
            initial={{ opacity: 0, y: 50, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 50, scale: 0.95 }}
            className="card-elevated"
            style={{
              position: "fixed",
              top: "50%", left: "50%",
              transform: "translate(-50%, -50%)",
              zIndex: 9999,
              width: "90%",
              maxWidth: "600px",
              padding: "2rem",
              display: "flex", flexDirection: "column", gap: "1.5rem"
            }}
          >
            <button 
              onClick={onClose} 
              style={{ position: "absolute", top: "1rem", right: "1rem", background: "none", border: "none", cursor: "pointer", color: "var(--text-secondary)" }}
            >
              <X size={24} />
            </button>

            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid var(--border-subtle)", paddingBottom: "1rem" }}>
              <div>
                <span className="text-meta" style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                  <FileText size={14} style={{ color: "var(--wine-primary)"}} /> ID: {data.label}
                </span>
                <div style={{ fontSize: "1.2rem", fontWeight: 800, marginTop: "0.25rem" }}>{data.metric}</div>
              </div>
              <span className={`badge ${data.risk === 'HIGH' || data.risk === 'CRITICAL' ? 'badge-rose' : 'badge-amber'}`} style={{ padding: "0.5rem 1rem", fontSize: "0.8rem" }}>
                {data.risk}
              </span>
            </div>

            <div style={{ display: "flex", alignItems: "flex-start", gap: "12px", background: "var(--wine-subtle)", padding: "1rem", borderRadius: "12px", border: "1px solid rgba(114, 47, 55, 0.2)" }}>
              <Info size={20} style={{ color: "var(--wine-primary)", flexShrink: 0 }} />
              <div style={{ color: "var(--wine-primary)", fontWeight: 500, fontSize: "0.95rem" }}>
                {summary}
              </div>
            </div>

            <div>
              <div className="text-meta" style={{ marginBottom: "1rem" }}>Top Contributing Factors (SHAP Values)</div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "1rem" }}>
                {factors.map((factor, j) => (
                  <div key={j} style={{ background: "rgba(114,47,55,0.03)", border: "1px solid rgba(114,47,55,0.1)", borderRadius: "10px", padding: "1rem" }}>
                    <div style={{ fontSize: "1.4rem", fontWeight: 800, color: "var(--wine-primary)", marginBottom: "0.25rem" }}>+{factor.shap}</div>
                    <div style={{ fontWeight: 600, fontSize: "0.85rem", color: "var(--text-primary)" }}>{factor.label}</div>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
