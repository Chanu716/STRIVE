"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";

export function LiveMap({ height = "400px" }: { height?: string }) {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<maplibregl.Map | null>(null);
  const isMapLoaded = useRef(false);

  const [origin, setOrigin] = useState<[number, number] | null>(null);
  const [destination, setDestination] = useState<[number, number] | null>(null);
  const [routes, setRoutes] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);
  const [alpha, setAlpha] = useState(0.5);

  const originMarker = useRef<maplibregl.Marker | null>(null);
  const destMarker = useRef<maplibregl.Marker | null>(null);

  // Keep refs in sync so click handler always sees fresh state
  const originRef = useRef(origin);
  const destRef = useRef(destination);
  useEffect(() => { originRef.current = origin; }, [origin]);
  useEffect(() => { destRef.current = destination; }, [destination]);

  // ── 1. Initialise map once ───────────────────────────────────────────────
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
      center: [-118.48, 34.02], // Santa Monica Area (where data is available)
      zoom: 12,
      attributionControl: false,
    });

    map.current.addControl(new maplibregl.NavigationControl({ showCompass: false }), "bottom-right");

    map.current.on("load", () => {
      isMapLoaded.current = true;
    });

    // Single click handler using refs — no stale closures
    map.current.on("click", (e) => {
      const coords: [number, number] = [e.lngLat.lng, e.lngLat.lat];
      if (!originRef.current) {
        setOrigin(coords);
      } else if (!destRef.current) {
        setDestination(coords);
      } else {
        // Third click — reset
        setOrigin(coords);
        setDestination(null);
        setRoutes([]);
        setStatusMsg(null);
      }
    });

    return () => {
      map.current?.remove();
      map.current = null;
      isMapLoaded.current = false;
    };
  }, []);

  // ── 2. Sync markers whenever origin / destination change ─────────────────
  useEffect(() => {
    if (!map.current) return;

    if (origin) {
      if (!originMarker.current) {
        originMarker.current = new maplibregl.Marker({ color: "#10b981" })
          .setLngLat(origin)
          .addTo(map.current);
      } else {
        originMarker.current.setLngLat(origin);
      }
    } else {
      originMarker.current?.remove();
      originMarker.current = null;
    }

    if (destination) {
      if (!destMarker.current) {
        destMarker.current = new maplibregl.Marker({ color: "#e11d48" })
          .setLngLat(destination)
          .addTo(map.current);
      } else {
        destMarker.current.setLngLat(destination);
      }
    } else {
      destMarker.current?.remove();
      destMarker.current = null;
    }
  }, [origin, destination]);

  // ── 3. Fetch routes from backend ─────────────────────────────────────────
  const fetchRoutes = useCallback(async () => {
    if (!origin || !destination) return;
    setIsLoading(true);
    setStatusMsg("Evaluating routes...");
    try {
      const res = await fetch("http://localhost:8000/v1/route/safe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          origin: { lon: origin[0], lat: origin[1] },
          destination: { lon: destination[0], lat: destination[1] },
          alpha: alpha,
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        setStatusMsg(`Error: ${err.detail || "Routing failed. Ensure points are on the road network."}`);
        return;
      }
      const data = await res.json();
      if (data.alternatives?.length) {
        setRoutes(data.alternatives);
        setStatusMsg(`${data.alternatives.length} route(s) found. Hover a line for details.`);
      } else {
        setStatusMsg("No routes returned by the server.");
      }
    } catch {
      setStatusMsg("Network error — is the backend running on port 8000?");
    } finally {
      setIsLoading(false);
    }
  }, [origin, destination, alpha]);

  // ── 4. Draw / update route layers whenever routes change ─────────────────
  useEffect(() => {
    const m = map.current;
    if (!m || routes.length === 0) return;

    const geojson: GeoJSON.FeatureCollection = {
      type: "FeatureCollection",
      features: routes.map((r, i) => ({
        type: "Feature",
        geometry: r.geometry,
        properties: {
          route_id: r.route_id ?? `route_${i}`,
          risk_score: r.avg_risk_score,
          distance_km: r.distance_km,
          duration_min: r.duration_min,
          is_safest: r.is_safest ? 1 : 0,
        },
      })),
    };

    const applyLayers = () => {
      const src = m.getSource("routes") as maplibregl.GeoJSONSource | undefined;
      if (src) {
        src.setData(geojson);
      } else {
        m.addSource("routes", { type: "geojson", data: geojson });
      }

      if (!m.getLayer("route-lines")) {
        m.addLayer({
          id: "route-lines",
          type: "line",
          source: "routes",
          layout: { "line-cap": "round", "line-join": "round" },
          paint: {
            "line-color": ["case", ["==", ["get", "is_safest"], 1], "#10b981", "#94a3b8"],
            "line-width": ["case", ["==", ["get", "is_safest"], 1], 7, 4],
            "line-opacity": ["case", ["==", ["get", "is_safest"], 1], 0.95, 0.65],
          },
        });
      }

      if (!m.getLayer("route-hover-target")) {
        m.addLayer({
          id: "route-hover-target",
          type: "line",
          source: "routes",
          paint: { "line-width": 24, "line-opacity": 0 },
        });

        const popup = new maplibregl.Popup({
          closeButton: false,
          closeOnClick: false,
          maxWidth: "260px",
        });

        m.on("mouseenter", "route-hover-target", (e) => {
          if (!e.features?.length) return;
          m.getCanvas().style.cursor = "pointer";
          const p = e.features[0].properties as any;
          const safeBadge = p.is_safest === 1
            ? `<span style="background:rgba(16,185,129,.15);color:#10b981;padding:2px 7px;border-radius:4px;font-size:11px;font-weight:700;margin-left:8px;">★ SAFEST</span>`
            : "";
          popup
            .setLngLat(e.lngLat)
            .setHTML(`
              <div style="font-family:system-ui,sans-serif;padding:4px 2px;">
                <div style="display:flex;align-items:center;margin-bottom:6px;">
                  <span style="font-weight:800;font-size:1rem;color:${p.is_safest === 1 ? "#10b981" : "#1e293b"}">
                    Risk: ${p.risk_score}/100
                  </span>${safeBadge}
                </div>
                <div style="font-size:.82rem;color:#64748b;line-height:1.6;">
                  📍 ${p.distance_km} km &nbsp;|&nbsp; ⏱ ${p.duration_min} min
                </div>
              </div>`)
            .addTo(m);
        });

        m.on("mouseleave", "route-hover-target", () => {
          m.getCanvas().style.cursor = "";
          popup.remove();
        });
      }
    };

    if (isMapLoaded.current) {
      applyLayers();
    } else {
      m.once("load", applyLayers);
    }

    try {
      const lngs = routes.flatMap((r) => r.geometry.coordinates.map((c: number[]) => c[0]));
      const lats = routes.flatMap((r) => r.geometry.coordinates.map((c: number[]) => c[1]));
      if (lngs.length && lats.length) {
        m.fitBounds(
          [[Math.min(...lngs), Math.min(...lats)], [Math.max(...lngs), Math.max(...lats)]],
          { padding: 60 }
        );
      }
    } catch { /* ignore */ }
  }, [routes]);

  return (
    <div style={{ position: "relative", width: "100%", height, borderRadius: "20px", overflow: "hidden" }}>
      <div ref={mapContainer} style={{ width: "100%", height: "100%" }} />

      <div style={{
        position: "absolute", top: 16, left: 16, zIndex: 10,
        background: "rgba(255,255,255,0.97)", backdropFilter: "blur(16px)",
        padding: "1.25rem 1.5rem", borderRadius: "16px",
        boxShadow: "0 8px 32px rgba(0,0,0,0.12)", border: "1px solid rgba(0,0,0,0.06)",
        width: 320,
      }}>
        <h4 style={{ margin: "0 0 10px", fontWeight: 800, fontSize: "1.1rem", color: "#1e293b" }}>
          STRIVE Global Safety Router
        </h4>

        <div style={{ position: "relative", marginBottom: "1rem" }}>
          <input
            type="text"
            placeholder="Search city (e.g. Delhi, London)..."
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                const query = (e.target as HTMLInputElement).value;
                fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}`)
                  .then(r => r.json())
                  .then(data => {
                    if (data?.length) {
                      const { lat, lon } = data[0];
                      map.current?.flyTo({ center: [parseFloat(lon), parseFloat(lat)], zoom: 12 });
                    }
                  });
              }
            }}
            style={{
              width: "100%", padding: "10px 12px", borderRadius: "10px",
              border: "1px solid #e2e8f0", fontSize: "0.85rem", outline: "none"
            }}
          />
        </div>

        <div style={{ marginBottom: "1.25rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
            <span style={{ fontSize: "0.75rem", fontWeight: 700, color: "#64748b" }}>Shortest Path</span>
            <span style={{ fontSize: "0.75rem", fontWeight: 700, color: "#64748b" }}>Safest Path</span>
          </div>
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={alpha}
            onChange={(e) => setAlpha(parseFloat(e.target.value))}
            style={{ width: "100%", accentColor: "#722F37", cursor: "pointer" }}
          />
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 10, marginBottom: 14 }}>
          {[
            { label: "Origin", val: origin, color: "#10b981" },
            { label: "Destination", val: destination, color: "#e11d48" },
          ].map(({ label, val, color }) => (
            <div key={label} style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <div style={{ width: 10, height: 10, borderRadius: "50%", background: color, flexShrink: 0 }} />
              <div style={{ flexGrow: 1 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <div style={{ fontWeight: 700, fontSize: "0.8rem", color: "#1e293b" }}>{label}</div>
                  {label === "Origin" && (
                    <button
                      onClick={() => {
                        navigator.geolocation.getCurrentPosition((pos) => {
                          const coords: [number, number] = [pos.coords.longitude, pos.coords.latitude];
                          setOrigin(coords);
                          map.current?.flyTo({ center: coords, zoom: 14 });
                        });
                      }}
                      style={{
                        background: "none", border: "none", color: "#6366f1",
                        fontSize: "0.68rem", fontWeight: 700, cursor: "pointer", padding: 0
                      }}
                    >
                      USE MY LOCATION
                    </button>
                  )}
                </div>
                <div style={{ fontSize: "0.75rem", color: "#64748b" }}>
                  {val ? `${val[1].toFixed(5)}, ${val[0].toFixed(5)}` : "Awaiting map click…"}
                </div>
              </div>
            </div>
          ))}
        </div>

        {statusMsg && (
          <div style={{
            fontSize: "0.75rem", color: "#475569", background: "#f1f5f9",
            borderRadius: 8, padding: "6px 10px", marginBottom: 12, lineHeight: 1.5,
          }}>
            {statusMsg}
          </div>
        )}

        <button
          id="compute-routes-btn"
          onClick={fetchRoutes}
          disabled={!origin || !destination || isLoading}
          style={{
            width: "100%", padding: "0.75rem",
            background: (!origin || !destination || isLoading) ? "#e2e8f0" : "#722F37",
            color: (!origin || !destination || isLoading) ? "#94a3b8" : "#fff",
            border: "none", borderRadius: 10, fontWeight: 800, fontSize: "0.88rem",
            cursor: (!origin || !destination || isLoading) ? "not-allowed" : "pointer",
            transition: "all 0.2s",
          }}
        >
          {isLoading ? "EVALUATING…" : "COMPUTE SAFE ROUTES"}
        </button>
      </div>
    </div>
  );
}
