"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { motion, Variants } from "framer-motion";
import { TrendingUp, ShieldAlert, Zap, Clock, Activity, Target, Map as MapIcon, LogOut } from "lucide-react";
import { BorderGlow } from "@/components/ui/BorderGlow";
import { LiveMap } from "@/components/LiveMap";
import { ShapModal } from "@/components/ShapModal";

const fadeUp: Variants = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 300, damping: 24 } }
};

const container: Variants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.1 }
  }
};

const stats = [
  { label: "Network Coverage", value: "98.2%", sub: "Full Vijayawada OSM density", icon: Activity, color: "var(--wine-primary)", bg: "var(--wine-subtle)" },
  { label: "Safety Accuracy", value: "94.1%", sub: "Verified risk predictions", icon: Target, color: "var(--accent-emerald)", bg: "rgba(5, 150, 105, 0.1)" },
  { label: "Live Sensors", value: "12", sub: "Active weather & traffic feeds", icon: Zap, color: "var(--accent-amber)", bg: "rgba(217, 119, 6, 0.1)" },
  { label: "Inference Latency", value: "38ms", sub: "Per-segment scoring speed", icon: Clock, color: "var(--accent-rose)", bg: "rgba(225, 29, 72, 0.1)" },
];

export default function AdminOverview() {
  const router = useRouter();
  const [authChecked, setAuthChecked] = useState(false);
  const [selectedShap, setSelectedShap] = useState<any>(null);
  const [activeRoute, setActiveRoute] = useState<any>(null);
  const [availableRoutes, setAvailableRoutes] = useState<any[]>([]);

  useEffect(() => {
    const isAuth = localStorage.getItem("strive_auth");
    if (!isAuth) {
      router.push("/login");
    } else {
      setAuthChecked(true);
    }
  }, [router]);

  const handleLogout = () => {
    localStorage.removeItem("strive_auth");
    router.push("/login");
  };

  if (!authChecked) return null;

  return (
    <div style={{ display: "flex", flexDirection: "column", minHeight: "100vh", background: "var(--bg-base)" }}>
      <header style={{ 
        display: "flex", justifyContent: "space-between", alignItems: "center", 
        padding: "0.75rem 2rem", background: "rgba(252, 251, 251, 0.9)", 
        backdropFilter: "blur(12px)", borderBottom: "1px solid var(--border-subtle)",
        position: "sticky", top: 0, zIndex: 100
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <img src="/STRIVE.png" alt="STRIVE Logo" style={{ height: "32px", objectFit: "contain" }} />
          <span style={{ fontWeight: 800, fontSize: "1.2rem", color: "var(--wine-primary)", letterSpacing: "-0.03em" }}>STRIVE</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "2rem" }}>
           <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <div className="animate-pulse" style={{ width: "8px", height: "8px", borderRadius: "50%", background: "var(--accent-emerald)", boxShadow: "0 0 10px rgba(5,150,105,0.6)" }} />
              <span className="text-meta" style={{ fontSize: "0.7rem" }}>CORE ENGINE ONLINE</span>
           </div>
           <button onClick={handleLogout} className="text-wine text-meta" style={{ display: "flex", alignItems: "center", gap: "6px", background: "none", border: "none", cursor: "pointer", fontWeight: 700, fontSize: "0.7rem" }}>
             <LogOut size={14} /> SIGN OUT
           </button>
        </div>
      </header>

      <main style={{ display: "grid", gridTemplateColumns: "350px 1fr", height: "calc(100vh - 60px)", overflow: "hidden" }}>
        <aside style={{ 
          background: "#fff", borderRight: "1px solid var(--border-subtle)", 
          display: "flex", flexDirection: "column", overflowY: "auto", padding: "1.5rem"
        }}>
          <div style={{ marginBottom: "2rem" }}>
            <div className="text-meta" style={{ marginBottom: "0.25rem" }}>Intelligence Portal</div>
            <h2 style={{ fontSize: "1.25rem", fontWeight: 900, margin: 0 }}>Command Center</h2>
          </div>

          <ShapModal 
            isOpen={!!selectedShap} 
            onClose={() => setSelectedShap(null)} 
            data={selectedShap} 
          />
          
          <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
            <div className="card-elevated" style={{ padding: "1.25rem", background: "var(--wine-subtle)", border: "none" }}>
               <h3 style={{ fontSize: "0.9rem", fontWeight: 800, marginBottom: "1rem", color: "var(--wine-primary)", display: "flex", alignItems: "center", gap: "8px" }}>
                 <MapIcon size={16}/> ROUTE PARAMETERS
               </h3>
               <div id="external-controls-portal" />
            </div>

            {availableRoutes.length > 0 && (
              <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
                <div className="text-meta">Available Pathways</div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "8px" }}>
                  {availableRoutes.map((r, i) => (
                    <button 
                      key={i}
                      onClick={() => setActiveRoute(r)}
                      style={{
                        padding: "0.75rem 0.5rem", borderRadius: "12px", border: "1px solid var(--border-subtle)",
                        background: activeRoute?.route_id === r.route_id ? "var(--wine-primary)" : "#fff",
                        color: activeRoute?.route_id === r.route_id ? "#fff" : "var(--text-primary)",
                        fontSize: "0.7rem", fontWeight: 800, cursor: "pointer", transition: "all 0.2s"
                      }}
                    >
                      {r.risk_tier.toUpperCase()}
                      <div style={{ fontSize: "0.6rem", opacity: 0.8, marginTop: "2px" }}>{r.avg_risk_score}/100</div>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* System Triggers */}
            <motion.div variants={fadeUp} className="card-elevated" style={{ padding: "1.25rem", flexGrow: 1, display: "flex", flexDirection: "column" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <h4 style={{ margin: 0, fontWeight: 800, fontSize: "1rem", color: "#1e293b" }}>
                  Live Risk Triggers
                </h4>
                <div style={{ padding: "4px", background: "var(--wine-subtle)", borderRadius: "6px" }}>
                  <Zap size={16} style={{ color: "var(--wine-primary)" }} />
                </div>
              </div>
              
              <div style={{ display: "flex", flexDirection: "column", gap: "12px", marginTop: "1rem" }}>
                {activeRoute?.top_factors?.length > 0 ? (
                  <>
                    {activeRoute.top_factors.map((factor: any, i: number) => (
                      <div key={i} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "0.75rem", background: "rgba(114,47,55,0.03)", borderRadius: "10px", border: "1px solid rgba(114,47,55,0.08)" }}>
                        <span style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--text-secondary)" }}>{factor.label}</span>
                        <span style={{ fontSize: "0.85rem", fontWeight: 800, color: "var(--wine-primary)" }}>+{factor.shap}</span>
                      </div>
                    ))}
                    <button 
                      onClick={() => setSelectedShap({
                        title: "Route Analysis",
                        risk: activeRoute.avg_risk_score > 60 ? "HIGH" : activeRoute.avg_risk_score > 40 ? "MEDIUM" : "LOW",
                        label: activeRoute.route_id,
                        metric: `Avg Risk: ${activeRoute.avg_risk_score}/100`,
                        factors: activeRoute.top_factors,
                        summary: activeRoute.summary
                      })}
                      style={{ marginTop: "12px", padding: "10px", borderRadius: "10px", border: "1px solid var(--wine-primary)", color: "var(--wine-primary)", background: "none", fontSize: "0.75rem", fontWeight: 700, cursor: "pointer" }}
                    >
                      VIEW DETAILED ANALYSIS
                    </button>
                  </>
                ) : (
                  <div style={{ textAlign: "center", padding: "2rem", color: "#94a3b8", fontSize: "0.8rem" }}>
                    Select a route to view risk factor analysis.
                  </div>
                )}
              </div>
            </motion.div>

            {/* Strategic Summary */}
            <motion.div variants={fadeUp} className="card-elevated" style={{ padding: "1.25rem", background: "rgba(255,255,255,0.8)" }}>
              {activeRoute ? (
                <div style={{ padding: "1rem", background: "var(--wine-subtle)", borderRadius: "12px", border: "1px solid rgba(114, 47, 55, 0.1)" }}>
                  <div style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--wine-primary)", marginBottom: "4px" }}>Strategic Summary</div>
                  <div style={{ fontSize: "0.75rem", color: "var(--wine-primary)", lineHeight: 1.4, fontWeight: 500 }}>
                    {activeRoute.summary}
                  </div>
                </div>
              ) : (
                <div style={{ fontSize: "0.75rem", color: "#94a3b8", textAlign: "center", padding: "10px" }}>
                  Risk analysis unavailable.
                </div>
              )}
            </motion.div>
          </div>
        </aside>

        <section style={{ position: "relative", background: "#e2e8f0" }}>
          <LiveMap 
            height="100%" 
            onRoutesFound={(r: any) => { setAvailableRoutes(r); setActiveRoute(r[0]); }} 
            activeRouteId={activeRoute?.route_id}
          />
          
          <div style={{ 
            position: "absolute", bottom: "2rem", left: "2rem", right: "2rem",
            display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "1rem",
            pointerEvents: "none"
          }}>
            {stats.map((stat, i) => {
              const Icon = stat.icon;
              return (
                <div key={i} style={{ 
                  background: "rgba(255,255,255,0.9)", backdropFilter: "blur(12px)",
                  padding: "1rem", borderRadius: "16px", border: "1px solid rgba(255,255,255,0.2)",
                  boxShadow: "0 4px 20px rgba(0,0,0,0.08)"
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "0.25rem" }}>
                    <Icon size={14} style={{ color: stat.color }} />
                    <span className="text-meta" style={{ fontSize: "0.65rem" }}>{stat.label}</span>
                  </div>
                  <div style={{ fontSize: "1.1rem", fontWeight: 900 }}>{stat.value}</div>
                </div>
              );
            })}
          </div>
        </section>
      </main>
    </div>
  );
}
