"use client";

import { motion } from "framer-motion";
import { ReactNode } from "react";

export function BorderGlow({
  children,
  glowColor = "var(--wine-glow)",
  className = ""
}: {
  children: ReactNode;
  glowColor?: string;
  className?: string;
}) {
  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }} className={className}>
      <motion.div
        animate={{ opacity: [0.5, 1, 0.5] }}
        transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
        style={{
          position: "absolute",
          inset: "-2px",
          background: glowColor,
          filter: "blur(12px)",
          borderRadius: "24px",
          zIndex: -1,
        }}
      />
      {children}
    </div>
  );
}
