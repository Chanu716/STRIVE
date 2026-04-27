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
  { label: "Segments Scored", value: "84,023", sub: "Live OpenStreetMap tracking", icon: Activity, color: "var(--wine-primary)", bg: "var(--wine-subtle)" },
  { label: "Avg Network Risk", value: "24/100", sub: "Currently LOW operating risk", icon: Target, color: "var(--accent-emerald)", bg: "rgba(5, 150, 105, 0.1)" },
  { label: "Active Weather Hazards", value: "3", sub: "Rain / Reduced visibility zones", icon: Zap, color: "var(--accent-amber)", bg: "rgba(217, 119, 6, 0.1)" },
  { label: "High Risk Segments", value: "112", sub: "Score > 75 based on live weather", icon: ShieldAlert, color: "var(--accent-rose)", bg: "rgba(225, 29, 72, 0.1)" },
];

export default function AdminOverview() {
  const router = useRouter();
  const [authChecked, setAuthChecked] = useState(false);
  const [selectedShap, setSelectedShap] = useState<any>(null);

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
    <div style={{ display: "flex", flexDirection: "column", minHeight: "100vh" }}>
      <header style={{ 
        display: "flex", justifyContent: "space-between", alignItems: "center", 
        padding: "1rem 2rem", background: "rgba(252, 251, 251, 0.9)", 
        backdropFilter: "blur(12px)", borderBottom: "1px solid var(--border-subtle)",
        position: "sticky", top: 0, zIndex: 100
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <img src="/STRIVE.png" alt="STRIVE Logo" style={{ height: "40px", objectFit: "contain" }} />
          <span style={{ fontWeight: 800, fontSize: "1.4rem", color: "var(--wine-primary)", letterSpacing: "-0.03em" }}>STRIVE</span>
        </div>
        <button onClick={handleLogout} className="text-wine text-meta" style={{ display: "flex", alignItems: "center", gap: "6px", background: "none", border: "none", cursor: "pointer", fontWeight: 700 }}>
          <LogOut size={16} /> SIGN OUT
        </button>
      </header>

      <div style={{ display: "flex", flexDirection: "column", gap: "2rem", padding: "2rem", maxWidth: "1600px", margin: "0 auto", width: "100%" }}>
        <ShapModal isOpen={!!selectedShap} onClose={() => setSelectedShap(null)} segmentData={selectedShap} />
        <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end" }}>
          <div>
            <div className="text-meta" style={{ marginBottom: "0.5rem" }}>Live Command Center</div>
            <h1 className="heading-hero">Network Overview</h1>
            <p style={{ color: "var(--text-secondary)", marginTop: "0.5rem", fontWeight: 500 }}>
              Real-time route analytics and risk zone monitoring.
            </p>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
            <span className="badge badge-emerald" style={{ display: "flex", alignItems: "center", gap: "6px" }}>
              <span style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--accent-emerald)", boxShadow: "0 0 8px var(--accent-emerald)" }} />
              LIVE
            </span>
          </div>
        </motion.div>

      <motion.div variants={container} initial="hidden" animate="show" style={{
        display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "1.5rem"
      }}>
        {stats.map((stat, i) => {
          const Icon = stat.icon;
          return (
            <motion.div key={i} variants={fadeUp} className="card-elevated hover-card" style={{ padding: "1.5rem", position: "relative", overflow: "hidden" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "1.5rem" }}>
                <div>
                  <div className="text-meta" style={{ marginBottom: "0.5rem" }}>{stat.label}</div>
                  <div style={{ fontSize: "2rem", fontWeight: 900, color: "var(--text-primary)", letterSpacing: "-0.03em" }}>{stat.value}</div>
                </div>
                <div style={{ width: "48px", height: "48px", borderRadius: "16px", background: stat.bg, display: "flex", alignItems: "center", justifyContent: "center" }}>
                  <Icon style={{ color: stat.color }} size={24} />
                </div>
              </div>
              <div style={{ fontSize: "0.85rem", color: "var(--text-secondary)", fontWeight: 500 }}>
                {stat.sub}
              </div>
            </motion.div>
          );
        })}
      </motion.div>

      <div style={{ display: "grid", gridTemplateColumns: "1.5fr 1fr", gap: "1.5rem", marginTop: "1rem" }}>
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }} className="card-base" style={{ padding: "2rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "2rem" }}>
            <div>
              <div className="text-meta" style={{ marginBottom: "0.5rem" }}>Geospatial Stream</div>
              <h2 style={{ fontSize: "1.5rem", fontWeight: 800, margin: 0 }}>Live Routing Map</h2>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <div className="animate-pulse" style={{ width: "8px", height: "8px", borderRadius: "50%", background: "var(--accent-emerald)", boxShadow: "0 0 10px rgba(5,150,105,0.6)" }} />
              <span className="text-meta">Live</span>
            </div>
          </div>
          
          <BorderGlow glowColor="var(--wine-glow)">
            <LiveMap height="400px" />
          </BorderGlow>
        </motion.div>

        <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.5 }} className="card-base" style={{ padding: "2rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "2rem" }}>
            <div>
              <div className="text-meta" style={{ marginBottom: "0.5rem" }}>Alerts</div>
              <h2 style={{ fontSize: "1.5rem", fontWeight: 800, margin: 0 }}>System Triggers</h2>
            </div>
            <div style={{ padding: "0.5rem", background: "var(--wine-subtle)", borderRadius: "10px" }}>
              <Zap size={18} style={{ color: "var(--wine-primary)" }} />
            </div>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
            {[
              { title: "Heavy Rain on Highway 101", risk: "HIGH", label: "way/123456789", metric: "Risk Score: 88/100" },
              { title: "Historical Accident Cluster", risk: "HIGH", label: "way/987654321", metric: "Risk: 74/100" },
              { title: "Reduced Visibility Zone", risk: "MEDIUM", label: "way/555555555", metric: "Risk: 42/100" }
            ].map((alert, i) => (
              <div key={i} style={{ padding: "1rem", borderRadius: "16px", background: "var(--bg-base)", border: "1px solid var(--border-subtle)" }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.5rem" }}>
                  <span style={{ fontWeight: 700, fontSize: "0.9rem" }}>{alert.title}</span>
                  <span className={`badge ${alert.risk === 'HIGH' ? 'badge-rose' : 'badge-amber'}`}>{alert.risk} RISK</span>
                </div>
                <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: "0.5rem" }}>Segment ID: {alert.label}</div>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginTop: "1rem" }}>
                   <span className="text-meta" style={{ display: "flex", alignItems: "center", gap: "4px" }}><Activity size={12}/> {alert.metric}</span>
                   <button onClick={() => setSelectedShap(alert)} className="text-wine text-meta" style={{ fontWeight: 800, background: "none", border: "none", cursor: "pointer" }}>VIEW SHAP ➔</button>
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </div>
    </div>
  );
}
