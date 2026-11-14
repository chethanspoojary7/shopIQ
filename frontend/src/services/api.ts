const BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5000';
async function get(path: string) {
  const r = await fetch(`${BASE}${path}`);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || `Request failed: ${r.status}`);
  return r.json();
}
export const api = {
  health: () => get('/api/health'),
  metadata: () => get('/api/metadata'),
  metrics: () => get('/api/metrics'),
  analytics: () => get('/api/analytics'),
  predict: async (session: any) => {
    const r = await fetch(`${BASE}/api/predict`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session }) });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.error || `Prediction failed: ${r.status}`);
    return j;
  },
};
export type SessionInput = {
  Administrative: number; Administrative_Duration: number; Informational: number;
  Informational_Duration: number; ProductRelated: number; ProductRelated_Duration: number;
  BounceRates: number; ExitRates: number; SpecialDay: number; Month: string;
  OperatingSystems: number; Browser: number; Region: number; TrafficType: number;
  VisitorType: string; Weekend: boolean;
};
export const DEFAULT_SESSION: SessionInput = {
  Administrative: 2, Administrative_Duration: 45, Informational: 1, Informational_Duration: 30,
  ProductRelated: 10, ProductRelated_Duration: 300, BounceRates: 0.02, ExitRates: 0.04,
  SpecialDay: 0, Month: 'May', OperatingSystems: 2, Browser: 2, Region: 1, TrafficType: 2,
  VisitorType: 'Returning_Visitor', Weekend: false,
};
