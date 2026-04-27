/**
 * Agent 1: Motion & UI Engine
 * Reusable framer-motion variants + animated wrappers
 * Premium dark-matte spring animations
 */

export const pageVariants = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { duration: 0.4, when: 'beforeChildren', staggerChildren: 0.07 },
  },
};

export const itemVariants = {
  hidden: { opacity: 0, y: 22, scale: 0.97 },
  show: {
    opacity: 1, y: 0, scale: 1,
    transition: { type: 'spring', stiffness: 280, damping: 28 },
  },
};

export const slideInLeft = {
  hidden: { opacity: 0, x: -30 },
  show: { opacity: 1, x: 0, transition: { type: 'spring', stiffness: 260, damping: 26 } },
};

export const slideInRight = {
  hidden: { opacity: 0, x: 30 },
  show: { opacity: 1, x: 0, transition: { type: 'spring', stiffness: 260, damping: 26 } },
};

export const fadeUp = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { duration: 0.45, ease: [0.23, 1, 0.32, 1] } },
};

export const scaleIn = {
  hidden: { opacity: 0, scale: 0.9 },
  show: { opacity: 1, scale: 1, transition: { type: 'spring', stiffness: 320, damping: 30 } },
};

/** KPI card hover spring */
export const kpiCardHover = {
  scale: 1.03,
  y: -3,
  boxShadow: '0 12px 40px rgba(0,0,0,0.6), 0 0 20px rgba(99,102,241,0.15)',
  transition: { type: 'spring', stiffness: 400, damping: 25 },
};

/** Segment card hover spring */
export const segCardHover = {
  scale: 1.015,
  y: -4,
  boxShadow: '0 16px 48px rgba(0,0,0,0.65)',
  transition: { type: 'spring', stiffness: 350, damping: 28 },
};

/** Chart card hover */
export const chartCardHover = {
  scale: 1.008,
  boxShadow: '0 8px 32px rgba(0,0,0,0.5)',
  transition: { type: 'spring', stiffness: 300, damping: 30 },
};

/** Stagger container */
export const staggerContainer = {
  hidden: {},
  show: { transition: { staggerChildren: 0.08, delayChildren: 0.05 } },
};

/** Tab switch */
export const tabContent = {
  hidden: { opacity: 0, y: 10 },
  show: { opacity: 1, y: 0, transition: { duration: 0.3, ease: 'easeOut' } },
  exit: { opacity: 0, y: -8, transition: { duration: 0.2 } },
};
