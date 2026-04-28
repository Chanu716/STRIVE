"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import ReactDOM from "react-dom";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";

// ── Tier config ───────────────────────────────────────────────────────────────
const TIER: Record<string, { color: string; label: string; width: number; opacity: number }> = {
  safest: { color: "#10b981", label: "✅ Safest",  width: 8,  opacity: 1.0 },
  medium: { color: "#f59e0b", label: "⚠️ Medium",  width: 6,  opacity: 0.9 },
  risky:  { color: "#ef4444", label: "🚨 Risky",   width: 5,  opacity: 0.8 },
};

interface LiveMapProps {
  height?: string;
  onRoutesFound?: (routes: any[]) => void;
  activeRouteId?: string | null;
}

export function LiveMap({ height = "400px", onRoutesFound, activeRouteId }: LiveMapProps) {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const isMapLoaded = useRef(false);

  const [origin, setOrigin] = useState<[number, number] | null>(null);
  const [destination, setDestination] = useState<[number, number] | null>(null);
  const [routes, setRoutes] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);
  const [alpha, setAlpha] = useState(0.5);
  const [portalNode, setPortalNode] = useState<HTMLElement | null>(null);

  const originMarker = useRef<maplibregl.Marker | null>(null);
  const destMarker = useRef<maplibregl.Marker | null>(null);

  const originRef = useRef(origin);
  const destRef = useRef(destination);
  useEffect(() => { originRef.current = origin; }, [origin]);
  useEffect(() => { destRef.current = destination; }, [destination]);

  useEffect(() => {
    setPortalNode(document.getElementById("external-controls-portal"));
  }, []);

  // ── 1. Init map ───────────────────────────────────────────────────────────
  useEffect(() => {
    if (map.current || !mapContainer.current) return;

    map.current = new maplibregl.Map({
      container: mapContainer.current,
      style: {
        version: 8,
        sources: {
          osm: {
            type: "raster",
            tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
            tileSize: 256,
            attribution: "© OpenStreetMap contributors",
          },
        },
        layers: [{ id: "osm", type: "raster", source: "osm", minzoom: 0, maxzoom: 19 }],
      },
      center: [80.648, 16.506], // Center on Vijayawada
      zoom: 12,
      attributionControl: false,
    });

    map.current.addControl(new maplibregl.NavigationControl({ showCompass: false }), "bottom-right");

    map.current.on("load", () => { isMapLoaded.current = true; });

    map.current.on("click", (e) => {
      const coords: [number, number] = [e.lngLat.lng, e.lngLat.lat];
      if (!originRef.current) {
        setOrigin(coords);
      } else if (!destRef.current) {
        setDestination(coords);
      } else {
        setOrigin(coords);
        setDestination(null);
        setRoutes([]);
        setStatusMsg(null);
      }
    });

    return () => { map.current?.remove(); map.current = null; isMapLoaded.current = false; };
  }, []);

  // ── 2. Markers ───────────────────────────────────────────────────────────
  useEffect(() => {
    if (!map.current) return;
    if (origin) {
      if (!originMarker.current) originMarker.current = new maplibregl.Marker({ color: "#10b981" }).setLngLat(origin).addTo(map.current);
      else originMarker.current.setLngLat(origin);
    } else { originMarker.current?.remove(); originMarker.current = null; }

    if (destination) {
      if (!destMarker.current) destMarker.current = new maplibregl.Marker({ color: "#e11d48" }).setLngLat(destination).addTo(map.current);
      else destMarker.current.setLngLat(destination);
    } else { destMarker.current?.remove(); destMarker.current = null; }
  }, [origin, destination]);

  // ── 3. Fetch routes ───────────────────────────────────────────────────────
  const fetchRoutes = useCallback(async () => {
    if (!origin || !destination) return;
    setIsLoading(true);
    setRoutes([]);
    setStatusMsg("⚡ Evaluating risk scores...");

    const slowTimer = setTimeout(() => {
      setStatusMsg("📡 New area detected — downloading map data...");
    }, 3000);

    try {
      const res = await fetch("http://localhost:8000/v1/route/safe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          origin:      { lon: origin[0],      lat: origin[1] },
          destination: { lon: destination[0], lat: destination[1] },
          alpha,
        }),
      });

      clearTimeout(slowTimer);

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        setStatusMsg(`❌ ${err.detail || "Routing failed."}`);
        return;
      }

      const data = await res.json();
      if (data.alternatives?.length) {
        setRoutes(data.alternatives);
        if (onRoutesFound) onRoutesFound(data.alternatives);
        setStatusMsg(`✅ ${data.alternatives.length} route(s) found.`);
      } else {
        setStatusMsg("⚠️ No routes returned.");
      }
    } catch {
      clearTimeout(slowTimer);
      setStatusMsg("❌ Network error — is the backend running?");
    } finally {
      setIsLoading(false);
    }
  }, [origin, destination, alpha, onRoutesFound]);

  // ── 4. Draw routes ───────────────────────────────────────────────────────
  useEffect(() => {
    const m = map.current;
    if (!m || routes.length === 0) return;

    const draw = () => {
      ["safest","medium","risky"].forEach(tier => {
        const sid = `route-${tier}`;
        if (m.getLayer(sid)) m.removeLayer(sid);
        if (m.getLayer(`${sid}-hover`)) m.removeLayer(`${sid}-hover`);
        if (m.getSource(sid)) m.removeSource(sid);
      });

      routes.forEach((r) => {
        const tier = (r.risk_tier as string) || "medium";
        const cfg  = TIER[tier] ?? TIER.medium;
        const srcId = `route-${tier}`;
        
        // Active highlight logic
        const isActive = activeRouteId === r.route_id;
        const finalColor = isActive ? cfg.color : "#94a3b8";
        const finalWidth = isActive ? cfg.width + 4 : cfg.width;
        const finalOpacity = isActive ? 1.0 : 0.4;

        if (!m.getSource(srcId)) {
          m.addSource(srcId, {
            type: "geojson",
            data: { type: "Feature", geometry: r.geometry, properties: { tier, id: r.route_id } },
          });
        }

        if (!m.getLayer(srcId)) {
          m.addLayer({
            id: srcId, type: "line", source: srcId,
            layout: { "line-cap": "round", "line-join": "round" },
            paint: { "line-color": finalColor, "line-width": finalWidth, "line-opacity": finalOpacity },
          }); 

          // Hover interaction
          const popup = new maplibregl.Popup({ closeButton: false, closeOnClick: false, offset: 15 });
          
          m.on("mouseenter", srcId, (e) => {
            m.getCanvas().style.cursor = "pointer";
            const props = r; // use current route data
            const html = `
              <div style="padding: 10px; font-family: Inter, sans-serif; min-width: 140px;">
                <div style="font-weight: 800; color: ${cfg.color}; font-size: 0.85rem; margin-bottom: 4px;">
                  ${cfg.label}
                </div>
                <div style="display: flex; flex-direction: column; gap: 4px; border-top: 1px solid #f1f5f9; pt: 4px; mt: 4px;">
                   <div style="font-size: 0.7rem; color: #64748b;">📏 <b>${props.distance_km} km</b></div>
                   <div style="font-size: 0.7rem; color: #64748b;">⏱️ <b>${props.duration_min} min</b></div>
                   <div style="font-size: 0.7rem; color: #64748b;">🛡️ Risk: <b>${props.avg_risk_score}/100</b></div>
                </div>
              </div>
            `;
            popup.setLngLat(e.lngLat).setHTML(html).addTo(m);
          });

          m.on("mouseleave", srcId, () => {
            m.getCanvas().style.cursor = "";
            popup.remove();
          });
        } else {
          m.setPaintProperty(srcId, "line-color", finalColor);
          m.setPaintProperty(srcId, "line-width", finalWidth);
          m.setPaintProperty(srcId, "line-opacity", finalOpacity);
        }
      });

      // Fit bounds if new routes arrive
      if (routes.length > 0 && !activeRouteId) {
        try {
          const lngs = routes.flatMap(r => r.geometry.coordinates.map((c: number[]) => c[0]));
          const lats = routes.flatMap(r => r.geometry.coordinates.map((c: number[]) => c[1]));
          m.fitBounds([[Math.min(...lngs), Math.min(...lats)], [Math.max(...lngs), Math.max(...lats)]], { padding: 50 });
        } catch {}
      }
    };

    if (isMapLoaded.current) draw(); else m.once("load", draw);
  }, [routes, activeRouteId]);

  const controlPanel = (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {/* City Search */}
      <input
        type="text"
        placeholder="Focus on city (e.g. Vijayawada)..."
        onKeyDown={(e) => {
          if (e.key !== "Enter") return;
          const q = (e.target as HTMLInputElement).value;
          fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(q)}`)
            .then(r => r.json())
            .then(d => { if (d?.length) map.current?.flyTo({ center: [parseFloat(d[0].lon), parseFloat(d[0].lat)], zoom: 13 }); });
        }}
        style={{ width: "100%", padding: "8px 12px", borderRadius: "8px", border: "1px solid #e2e8f0", fontSize: "0.75rem", outline: "none", boxSizing: "border-box" }}
      />

      {/* Safety Slider */}
      <div>
        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
          <span style={{ fontSize: "0.65rem", fontWeight: 700, color: "#64748b" }}>Speed</span>
          <span style={{ fontSize: "0.65rem", fontWeight: 700, color: "#64748b" }}>Safety</span>
        </div>
        <input type="range" min="0" max="1" step="0.1" value={alpha}
          onChange={(e) => setAlpha(parseFloat(e.target.value))}
          style={{ width: "100%", accentColor: "var(--wine-primary)", cursor: "pointer" }} />
      </div>

      {/* Pin display */}
      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {[
          { label: "Origin",      val: origin,      color: "#10b981" },
          { label: "Destination", val: destination, color: "#e11d48" },
        ].map(({ label, val, color }) => (
          <div key={label} style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{ width: 8, height: 8, borderRadius: "50%", background: color, flexShrink: 0 }} />
            <div style={{ flexGrow: 1 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ fontWeight: 700, fontSize: "0.7rem", color: "#1e293b" }}>{label}</span>
                {label === "Origin" && (
                  <button onClick={() => navigator.geolocation.getCurrentPosition((p) => {
                      const c: [number, number] = [p.coords.longitude, p.coords.latitude];
                      setOrigin(c); map.current?.flyTo({ center: c, zoom: 14 });
                    })}
                    style={{ background: "none", border: "none", color: "var(--wine-primary)", fontSize: "0.6rem", fontWeight: 700, cursor: "pointer", padding: 0 }}>
                    AUTO
                  </button>
                )}
              </div>
              <div style={{ fontSize: "0.65rem", color: "#64748b", fontFamily: "monospace" }}>
                {val ? `${val[1].toFixed(4)}, ${val[0].toFixed(4)}` : "Not set"}
              </div>
            </div>
          </div>
        ))}
      </div>

      {statusMsg && (
        <div style={{ fontSize: "0.65rem", color: "var(--wine-primary)", background: "var(--wine-subtle)", borderRadius: "8px", padding: "6px 8px", border: "1px solid rgba(114, 47, 55, 0.1)" }}>
          {statusMsg}
        </div>
      )}

      <button
        onClick={fetchRoutes}
        disabled={!origin || !destination || isLoading}
        style={{
          width: "100%", padding: "0.6rem",
          background: (!origin || !destination || isLoading) ? "#f1f5f9" : "var(--wine-primary)",
          color: (!origin || !destination || isLoading) ? "#94a3b8" : "#fff",
          border: "none", borderRadius: "8px", fontWeight: 800, fontSize: "0.75rem",
          cursor: (!origin || !destination || isLoading) ? "not-allowed" : "pointer",
          transition: "all 0.2s",
        }}>
        {isLoading ? "COMPUTING..." : "GENERATE ROUTES"}
      </button>
    </div>
  );

  return (
    <div style={{ position: "relative", width: "100%", height, overflow: "hidden" }}>
      <div ref={mapContainer} style={{ width: "100%", height: "100%" }} />
      {portalNode && ReactDOM.createPortal(controlPanel, portalNode)}
    </div>
  );
}
