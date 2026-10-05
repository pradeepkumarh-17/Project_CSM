// Use a concrete string value for stressLevel (no TypeScript union syntax in JS)
sessionStorage.setItem(
  "finalResult",
  JSON.stringify({
    stressLevel: "medium",
    confidence: 92.4,
    phase1: { stressLevel: "medium", confidence: 90.1 },
    phase2: { stressLevel: "high", confidence: 87.3 },
    phase3: { stressLevel: "low", confidence: 88.5 },
  })
);
