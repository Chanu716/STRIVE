"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { MapPin, ShieldAlert, KeyRound, Loader2 } from "lucide-react";

export default function LoginPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [locating, setLocating] = useState(false);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setErrorMsg("");

    // Simulate auth
    await new Promise((r) => setTimeout(r, 1000));
    
    // Request geolocation
    setLocating(true);
    if (!navigator.geolocation) {
      setErrorMsg("Geolocation is not supported by your browser.");
      setLoading(false);
      setLocating(false);
      return;
    }

    navigator.geolocation.getCurrentPosition(
      (position) => {
        // Success
        console.log("Location access granted:", position.coords);
        localStorage.setItem("strive_auth", "true");
        router.push("/dashboard");
      },
      (error) => {
        console.error("Geolocation error:", error);
        setErrorMsg("Location access is required for safety routing.");
        setLoading(false);
        setLocating(false);
      }
    );
  };

  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", minHeight: "100vh", position: "relative", overflow: "hidden" }}>
      {/* Background elements */}
      <div style={{ position: "absolute", top: "-10%", left: "-10%", width: "50%", height: "50%", background: "radial-gradient(circle, var(--wine-glow) 0%, transparent 70%)", opacity: 0.5, zIndex: -1 }} />
      <div style={{ position: "absolute", bottom: "-10%", right: "-10%", width: "50%", height: "50%", background: "radial-gradient(circle, var(--wine-glow) 0%, transparent 70%)", opacity: 0.5, zIndex: -1 }} />
      
      <motion.div 
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.4 }}
        className="card-base"
        style={{ width: "100%", maxWidth: "420px", padding: "3rem", display: "flex", flexDirection: "column", gap: "2rem", zIndex: 1 }}
      >
        <div style={{ textAlign: "center" }}>
          <div style={{ display: "inline-flex", padding: "12px", borderRadius: "16px", background: "var(--wine-subtle)", color: "var(--wine-primary)", marginBottom: "1rem" }}>
            <ShieldAlert size={32} />
          </div>
          <h1 className="heading-hero" style={{ fontSize: "2rem", marginBottom: "0.5rem" }}>STRIVE</h1>
          <p style={{ color: "var(--text-secondary)", fontWeight: 500 }}>Global Safety Routing Engine</p>
        </div>

        <form onSubmit={handleLogin} style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
          <div>
            <label style={{ display: "block", fontSize: "0.85rem", fontWeight: 700, color: "var(--text-secondary)", marginBottom: "0.5rem" }}>ACCESS KEY</label>
            <div style={{ position: "relative", display: "flex", alignItems: "center" }}>
              <KeyRound size={18} style={{ color: "var(--text-secondary)", position: "absolute", left: "1rem" }} />
              <input 
                type="password" 
                required
                placeholder="Enter organization token..."
                style={{ 
                  width: "100%", padding: "1rem 1rem 1rem 3rem", 
                  background: "var(--bg-base)", border: "1px solid var(--border-subtle)", 
                  borderRadius: "12px", color: "var(--text-primary)", fontSize: "1rem", outline: "none",
                  boxShadow: "inset 0 2px 4px rgba(0,0,0,0.02)"
                }} 
              />
            </div>
          </div>

          <AnimatePresence>
            {errorMsg && (
              <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }} style={{ color: "var(--accent-rose)", fontSize: "0.85rem", fontWeight: 600, textAlign: "center" }}>
                {errorMsg}
              </motion.div>
            )}
          </AnimatePresence>

          <button 
            type="submit" 
            disabled={loading}
            style={{ 
              marginTop: "1rem", padding: "1rem", width: "100%", 
              background: "var(--wine-primary)", color: "#fff", 
              border: "none", borderRadius: "12px", 
              fontSize: "1rem", fontWeight: 800, cursor: loading ? "not-allowed" : "pointer",
              display: "flex", justifyContent: "center", alignItems: "center", gap: "8px",
              boxShadow: "0 8px 16px var(--wine-glow)", transition: "all 0.2s"
            }}
          >
            {loading ? (
              locating ? <><MapPin className="animate-pulse" size={18} /> Requesting Location...</> : <><Loader2 className="animate-spin" size={18} /> Authenticating...</>
            ) : "SIGN IN"}
          </button>
        </form>

        <div style={{ textAlign: "center", fontSize: "0.8rem", color: "var(--text-secondary)", fontWeight: 500 }}>
          Location access is mandatory for STRIVE routing telemetry. By signing in, you agree to spatial tracking.
        </div>
      </motion.div>
    </div>
  );
}
