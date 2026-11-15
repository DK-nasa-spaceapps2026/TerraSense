"use client";

import { useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Edges, Html, OrthographicCamera, OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import type { CellState, Status, ZoneProps } from "@/lib/types";

export const STATUS_COLOR: Record<Status, string> = {
  lush: "#3f9b3a",
  moderate: "#8fb04c",
  water_stress: "#d4a73a",
  heat_stress: "#d6702b",
  bare_vulnerable: "#a9825c",
};
export const STATUS_LABEL: Record<Status, string> = {
  lush: "High vigor", moderate: "Moderate", water_stress: "Water stress",
  heat_stress: "Heat stress", bare_vulnerable: "Bare-soil risk",
};

// deterministic pseudo-random per tile so plants/cracks don't jitter between renders
const rnd = (seed: number) => { const x = Math.sin(seed * 127.1) * 43758.5453; return x - Math.floor(x); };

interface TileProps {
  zone: ZoneProps; cell: CellState; x: number; z: number;
  hovered: boolean; selected: boolean;
  onHover: (id: string | null) => void; onSelect: (id: string) => void;
}

function Tile({ zone, cell, x, z, hovered, selected, onHover, onSelect }: TileProps) {
  const body = useRef<THREE.Mesh>(null!);
  const mat = useRef<THREE.MeshStandardMaterial>(null!);
  const plants = useRef<THREE.Group>(null!);
  const cracks = useRef<THREE.Group>(null!);
  const sel = useRef<THREE.Group>(null!);
  const target = useMemo(() => new THREE.Color(), []);
  const seed = zone.row * 17 + zone.col * 31;

  const plantSpots = useMemo(() => Array.from({ length: 9 }, (_, i) => ({
    x: ((i % 3) - 1) * 0.28 + (rnd(seed + i) - 0.5) * 0.08,
    z: (Math.floor(i / 3) - 1) * 0.28 + (rnd(seed + i + 9) - 0.5) * 0.08,
  })), [seed]);
  const crackLines = useMemo(() => Array.from({ length: 5 }, (_, i) => ({
    x: (rnd(seed + i * 3) - 0.5) * 0.7, z: (rnd(seed + i * 5) - 0.5) * 0.7,
    len: 0.25 + rnd(seed + i) * 0.35, rot: rnd(seed + i * 7) * Math.PI,
  })), [seed]);

  useFrame((_, dt) => {
    const k = 1 - Math.exp(-dt * 6); // smooth critically-damped-ish transition
    const h = 0.22 + cell.vigor * 0.30 + (hovered ? 0.1 : 0) + (selected ? 0.06 : 0);
    body.current.scale.y += (h - body.current.scale.y) * k;
    body.current.position.y = body.current.scale.y / 2;
    // exposed soil darkens / dries toward the bare colour as risk rises
    target.set(STATUS_COLOR[cell.status]);
    mat.current.color.lerp(target, k);
    mat.current.emissive.setScalar(hovered ? 0.18 : selected ? 0.1 : 0);

    const top = body.current.scale.y;
    const ps = Math.max(0.001, (cell.ndvi - 0.2) * 1.8) * (cell.bare > 0.68 ? 0.3 : 1);
    plants.current.position.y = top;
    plants.current.scale.y += (ps - plants.current.scale.y) * k;
    plants.current.scale.x = plants.current.scale.z = 0.4 + 0.6 * Math.min(1, ps);
    cracks.current.position.y = top + 0.006;
    const cs = cell.bare > 0.62 ? 1 : 0;
    cracks.current.scale.setScalar(cracks.current.scale.x + (cs - cracks.current.scale.x) * k + 1e-4);
    cracks.current.visible = cracks.current.scale.x > 0.02;
    sel.current.visible = hovered || selected;
    sel.current.scale.y = top; sel.current.position.y = top / 2;
  });

  return (
    <group position={[x, 0, z]}>
      <mesh ref={body} castShadow receiveShadow
        onPointerOver={(e) => { e.stopPropagation(); onHover(zone.zone_id); document.body.style.cursor = "pointer"; }}
        onPointerOut={() => { onHover(null); document.body.style.cursor = "auto"; }}
        onClick={(e) => { e.stopPropagation(); onSelect(zone.zone_id); }}>
        <boxGeometry args={[0.94, 1, 0.94]} />
        <meshStandardMaterial ref={mat} color={STATUS_COLOR[cell.status]} roughness={0.9} />
      </mesh>

      <group ref={plants} scale={[1, 0.001, 1]}>
        {plantSpots.map((p, i) => (
          <mesh key={i} position={[p.x, 0.09, p.z]} scale={[1, 1, 1]}>
            <boxGeometry args={[0.1, 0.18, 0.1]} />
            <meshStandardMaterial color={cell.status === "lush" ? "#2c7a2c" : cell.status === "moderate" ? "#6f9a35" : "#b59a3a"} />
          </mesh>
        ))}
      </group>

      <group ref={cracks} scale={0.0001}>
        {crackLines.map((c, i) => (
          <mesh key={i} position={[c.x, 0, c.z]} rotation={[0, c.rot, 0]}>
            <boxGeometry args={[c.len, 0.008, 0.018]} />
            <meshBasicMaterial color="#4a3320" />
          </mesh>
        ))}
      </group>

      <group ref={sel} visible={false}>
        <mesh><boxGeometry args={[0.98, 1, 0.98]} /><meshBasicMaterial visible={false} />
          <Edges color={selected ? "#ffffff" : "#ffe9a8"} lineWidth={selected ? 3 : 1.5} />
        </mesh>
      </group>
    </group>
  );
}

export interface IsoGridProps {
  zones: ZoneProps[]; cells: CellState[]; rows: number; cols: number;
  hoveredId: string | null; selectedId: string | null;
  onHover: (id: string | null) => void; onSelect: (id: string | null) => void;
  locked?: boolean;
}

export default function IsoGrid({ zones, cells, rows, cols, hoveredId, selectedId, onHover, onSelect, locked = false }: IsoGridProps) {
  const [ready] = useState(true);
  const hoveredIdx = zones.findIndex((z) => z.zone_id === hoveredId);
  return (
    <Canvas shadows dpr={[1, 2]} onPointerMissed={() => onSelect(null)}>
      <color attach="background" args={["#141413"]} />
      <OrthographicCamera makeDefault position={[9, 9, 9]} zoom={Math.min(62, 330 / Math.max(rows, cols) * 1.1)} near={-50} far={100} />
      <ambientLight intensity={0.75} />
      <directionalLight position={[6, 10, 4]} intensity={1.6} castShadow shadow-mapSize={[1024, 1024]} />
      <OrbitControls enabled={!locked} enablePan={false} enableZoom={!locked} enableRotate={!locked} minZoom={30} maxZoom={120} minPolarAngle={0.7} maxPolarAngle={1.2} />

      {/* soil slab the tiles sit on — gives the "Civilization map chunk" look */}
      <mesh position={[0, -0.18, 0]} receiveShadow>
        <boxGeometry args={[cols + 0.25, 0.36, rows + 0.25]} />
        <meshStandardMaterial color="#3a2d20" roughness={1} />
      </mesh>

      {ready && zones.map((zone, i) => cells[i] && (
        <Tile key={zone.zone_id} zone={zone} cell={cells[i]}
          x={zone.col - (cols - 1) / 2} z={zone.row - (rows - 1) / 2}
          hovered={hoveredId === zone.zone_id} selected={selectedId === zone.zone_id}
          onHover={onHover} onSelect={onSelect} />
      ))}

      {hoveredIdx >= 0 && cells[hoveredIdx] && (
        <Html position={[zones[hoveredIdx].col - (cols - 1) / 2, 1.3, zones[hoveredIdx].row - (rows - 1) / 2]}
          center style={{ pointerEvents: "none" }}>
          <div className="whitespace-nowrap rounded bg-black/80 px-2 py-1 text-xs text-stone-100 shadow">
            {STATUS_LABEL[cells[hoveredIdx].status]} · SMI {cells[hoveredIdx].smi.toFixed(2)}
          </div>
        </Html>
      )}
    </Canvas>
  );
}
