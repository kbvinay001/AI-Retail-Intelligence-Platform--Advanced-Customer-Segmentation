/**
 * Agent 3: react-three-fiber Spatial Engine
 * Interactive 3D Supply Chain / Store Network
 * - Glowing pulsing store nodes
 * - Animated revenue flow lines
 * - Orbit controls, environment lighting
 * - React Suspense boundary with sleek loader
 */
import { Suspense, useRef, useState, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Stars, Sphere, Line } from '@react-three/drei';
import { motion, AnimatePresence } from 'framer-motion';
import * as THREE from 'three';

// ─── Store data (matches multi-store demo) ────────────────────────────────────
const STORES = [
  { id: 'MUM', name: 'Mumbai',    pos: [-2.5,  0.5,  0.5], color: '#6366f1', revenue: 107080, region: 'West'  },
  { id: 'DEL', name: 'Delhi',     pos: [ 1.5,  2.0,  0.0], color: '#22d3ee', revenue: 118691, region: 'North' },
  { id: 'BLR', name: 'Bangalore', pos: [ 0.5, -1.5,  1.0], color: '#a855f7', revenue: 114003, region: 'South' },
  { id: 'CHN', name: 'Chennai',   pos: [ 2.0, -2.0, -0.5], color: '#f59e0b', revenue: 114896, region: 'South' },
  { id: 'HYD', name: 'Hyderabad', pos: [ 1.0,  0.0,  2.0], color: '#34d399', revenue: 115325, region: 'South' },
  { id: 'HUB', name: 'HQ Hub',    pos: [-0.2,  0.2,  0.0], color: '#f97316', revenue: 0,       region: 'Core'  },
];

// ─── Pulsing glowing store node ───────────────────────────────────────────────
function StoreNode({ store, selected, onSelect }) {
  const meshRef   = useRef();
  const glowRef   = useRef();
  const ringRef   = useRef();
  const [hovered, setHovered] = useState(false);

  useFrame(({ clock }) => {
    const t = clock.elapsedTime;
    // Pulse scale
    const pulse = 1 + Math.sin(t * 2.5 + store.pos[0]) * 0.06;
    if (meshRef.current) meshRef.current.scale.setScalar(selected ? 1.35 * pulse : hovered ? 1.2 : pulse);
    // Glow breath
    if (glowRef.current) {
      glowRef.current.scale.setScalar(1.8 + Math.sin(t * 2 + store.pos[1]) * 0.3);
      glowRef.current.material.opacity = 0.08 + Math.sin(t * 2) * 0.04;
    }
    // Rotate ring
    if (ringRef.current) ringRef.current.rotation.z += 0.008;
  });

  const col = new THREE.Color(store.color);

  return (
    <group
      position={store.pos}
      onClick={() => onSelect(selected ? null : store)}
      onPointerOver={() => setHovered(true)}
      onPointerOut={() => setHovered(false)}
    >
      {/* Main sphere */}
      <mesh ref={meshRef}>
        <sphereGeometry args={[0.14, 32, 32]} />
        <meshStandardMaterial
          color={store.color}
          emissive={store.color}
          emissiveIntensity={hovered || selected ? 2.5 : 1.2}
          roughness={0.1}
          metalness={0.4}
        />
      </mesh>

      {/* Glow halo */}
      <mesh ref={glowRef}>
        <sphereGeometry args={[0.22, 16, 16]} />
        <meshBasicMaterial color={store.color} transparent opacity={0.08} side={THREE.BackSide} />
      </mesh>

      {/* Orbit ring */}
      <mesh ref={ringRef} rotation={[Math.PI / 2.2, 0, 0]}>
        <torusGeometry args={[0.22, 0.008, 8, 48]} />
        <meshBasicMaterial color={store.color} transparent opacity={selected ? 0.9 : 0.35} />
      </mesh>

      {/* Point light for local glow */}
      <pointLight color={store.color} intensity={selected ? 3 : 1.2} distance={1.2} />

      {/* Label */}
      <Text
        position={[0, 0.3, 0]}
        fontSize={0.1}
        color={hovered || selected ? '#ffffff' : '#94a3b8'}
        anchorX="center"
        anchorY="bottom"
        fontWeight={700}
      >
        {store.name}
      </Text>
      {store.revenue > 0 && (selected || hovered) && (
        <Text
          position={[0, 0.44, 0]}
          fontSize={0.075}
          color={store.color}
          anchorX="center"
        >
          ${(store.revenue / 1000).toFixed(0)}K
        </Text>
      )}
    </group>
  );
}

// ─── Animated edge / flow line ────────────────────────────────────────────────
function FlowLine({ from, to, color, active }) {
  const lineRef = useRef();
  const points  = useMemo(() => [
    new THREE.Vector3(...from),
    new THREE.Vector3(...to),
  ], [from, to]);

  useFrame(({ clock }) => {
    if (lineRef.current) {
      const t = (Math.sin(clock.elapsedTime * 1.5) + 1) / 2;
      if (lineRef.current.material) {
        lineRef.current.material.opacity = active ? 0.5 + t * 0.4 : 0.08 + t * 0.06;
      }
    }
  });

  return (
    <Line
      ref={lineRef}
      points={points}
      color={active ? color : '#2a2a45'}
      lineWidth={active ? 1.5 : 0.8}
      transparent
      opacity={active ? 0.7 : 0.1}
    />
  );
}

// ─── 3D Scene ─────────────────────────────────────────────────────────────────
function NetworkScene({ selected, onSelect }) {
  // Connect each store to HQ hub
  const HQ = STORES.find(s => s.id === 'HUB');

  return (
    <>
      {/* Background stars */}
      <Stars radius={12} depth={50} count={1200} factor={3} saturation={0.5} fade speed={1.2} />

      {/* Ambient + directional lighting */}
      <ambientLight intensity={0.4} />
      <directionalLight position={[5, 5, 5]} intensity={0.8} color="#ffffff" />
      <directionalLight position={[-5, -5, -5]} intensity={0.3} color="#6366f1" />

      {/* HQ to store edges */}
      {STORES.filter(s => s.id !== 'HUB').map(store => (
        <FlowLine
          key={store.id}
          from={HQ.pos}
          to={store.pos}
          color={store.color}
          active={selected?.id === store.id || selected?.id === 'HUB'}
        />
      ))}

      {/* Store nodes */}
      {STORES.map(store => (
        <StoreNode
          key={store.id}
          store={store}
          selected={selected?.id === store.id}
          onSelect={onSelect}
        />
      ))}

      <OrbitControls
        enableZoom={true}
        enablePan={false}
        autoRotate={!selected}
        autoRotateSpeed={0.5}
        minDistance={3}
        maxDistance={9}
      />
    </>
  );
}

// ─── Suspense loader ─────────────────────────────────────────────────────────
function Loader() {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      height: '100%', flexDirection: 'column', gap: '1rem',
    }}>
      <motion.div
        style={{
          width: 48, height: 48, borderRadius: '50%',
          border: '2px solid #1e1e35', borderTopColor: '#6366f1',
        }}
        animate={{ rotate: 360 }}
        transition={{ duration: 0.9, repeat: Infinity, ease: 'linear' }}
      />
      <div style={{ fontSize: '.78rem', color: 'var(--text3)' }}>Loading 3D scene…</div>
    </div>
  );
}

// ─── Public component ─────────────────────────────────────────────────────────
export default function StoreNetwork3D() {
  const [selected, setSelected] = useState(null);

  return (
    <motion.div
      className="chart-card"
      initial={{ opacity: 0, scale: 0.97 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ type: 'spring', stiffness: 220, damping: 26, delay: 0.2 }}
      style={{ position: 'relative', overflow: 'hidden' }}
    >
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
        <div>
          <div className="chart-card-title" style={{ marginBottom: '.2rem' }}>
            3D Store Network — Supply Chain View
          </div>
          <div style={{ fontSize: '.68rem', color: 'var(--text3)' }}>
            Click a node to inspect · Drag to orbit · Scroll to zoom
          </div>
        </div>
        {selected && (
          <AnimatePresence>
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              style={{
                background: '#13131f', border: '1px solid #2a2a45',
                borderRadius: 10, padding: '.6rem 1rem', fontSize: '.78rem',
                borderLeft: `3px solid ${selected.color}`,
              }}
            >
              <div style={{ color: selected.color, fontWeight: 700 }}>{selected.name}</div>
              {selected.revenue > 0 && (
                <div style={{ color: 'var(--text2)' }}>Revenue: ${selected.revenue.toLocaleString()}</div>
              )}
              <div style={{ color: 'var(--text3)', fontSize: '.68rem' }}>Region: {selected.region}</div>
            </motion.div>
          </AnimatePresence>
        )}
      </div>

      {/* Canvas */}
      <div style={{ height: 380, borderRadius: 10, overflow: 'hidden', background: '#080810' }}>
        <Suspense fallback={<Loader />}>
          <Canvas
            camera={{ position: [0, 0, 7], fov: 52 }}
            gl={{ antialias: true, alpha: false }}
            style={{ background: 'transparent' }}
          >
            <NetworkScene selected={selected} onSelect={setSelected} />
          </Canvas>
        </Suspense>
      </div>

      {/* Legend */}
      <div style={{ display: 'flex', gap: '.75rem', marginTop: '.85rem', flexWrap: 'wrap' }}>
        {STORES.map(s => (
          <button
            key={s.id}
            onClick={() => setSelected(prev => prev?.id === s.id ? null : s)}
            style={{
              display: 'flex', alignItems: 'center', gap: '.4rem',
              fontSize: '.7rem', background: selected?.id === s.id ? `${s.color}18` : 'transparent',
              border: `1px solid ${selected?.id === s.id ? s.color : '#1e1e35'}`,
              borderRadius: 99, padding: '.2rem .65rem', color: selected?.id === s.id ? s.color : '#64748b',
              cursor: 'pointer', transition: 'all .18s',
            }}
          >
            <span style={{ width: 6, height: 6, borderRadius: '50%', background: s.color, display: 'inline-block' }} />
            {s.name}
          </button>
        ))}
      </div>
    </motion.div>
  );
}
