import { useEffect, useMemo, useState } from 'react';
import { ArrowLeft, Bell, Calendar, ChevronDown, Filter, Plus, Search, Settings, Sparkles, Trash2 } from 'lucide-react';
import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { api, DEFAULT_SESSION, SessionInput } from './services/api';

type Tab = 'Overview' | 'Predict' | 'Models' | 'Insights';
const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
const fmt = (n: number) => n.toLocaleString('en-US');

function Field({ label, children }: any) { return <div><label>{label}</label>{children}</div>; }

export default function App() {
  const [tab, setTab] = useState<Tab>('Overview');
  const [analytics, setAnalytics] = useState<any>(null);
  const [metrics, setMetrics] = useState<any>(null);
  const [meta, setMeta] = useState<any>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const [visitorFilter, setVisitorFilter] = useState('All visitors');
  const [monthFilter, setMonthFilter] = useState('All months');
  const [detailModel, setDetailModel] = useState<string>('');
  const [form, setForm] = useState<SessionInput>(DEFAULT_SESSION);
  const [result, setResult] = useState<any>(null);
  const [predicting, setPredicting] = useState(false);
  const [history, setHistory] = useState<any[]>(() => { try { return JSON.parse(localStorage.getItem('shoppredict_history') || '[]'); } catch { return []; } });

  useEffect(() => {
    (async () => {
      try {
        const [a, m, md] = await Promise.all([api.analytics(), api.metrics(), api.metadata()]);
        setAnalytics(a); setMetrics(m); setMeta(md);
        setDetailModel(m.selected_model);
      } catch (e: any) { setError(e.message + ' — Is the Flask backend running on port 5000? Run: python -m backend.app'); }
      finally { setLoading(false); }
    })();
  }, []);

  const models = metrics?.models || [];
  const detail = models.find((m: any) => m.model === detailModel) || models[0];
  const monthData = useMemo(() => analytics ? ['Sep', 'Oct', 'Nov', 'Dec'].map(month => ({ month, rate: +(analytics.purchase_rate_by_month[month] as number).toFixed(3), pct: Math.round((analytics.purchase_rate_by_month[month] as number) * 100) })) : [], [analytics]);
  const visitorData = useMemo(() => analytics ? Object.entries(analytics.purchase_rate_by_visitor).map(([name, rate]) => ({ name, rate: +(rate as number).toFixed(3) })) : [], [analytics]);
  const importance = (metrics?.feature_importance || []).slice(0, 10);
  const filteredNote = visitorFilter !== 'All visitors' || monthFilter !== 'All months'
    ? `Filters shown for context: ${visitorFilter} · ${monthFilter}. Headline totals below are the full dataset (filters do not rewrite verified totals).`
    : 'Headline totals are the full verified dataset.';

  async function doPredict() {
    setPredicting(true); setResult(null);
    try {
      const r = await api.predict(form);
      setResult(r);
      const entry = { ...r, at: new Date().toLocaleString(), visitor: form.VisitorType, month: form.Month };
      const next = [entry, ...history].slice(0, 50);
      setHistory(next); localStorage.setItem('shoppredict_history', JSON.stringify(next));
    } catch (e: any) { setResult({ error: e.message }); }
    finally { setPredicting(false); }
  }

  if (loading) return <div className="shell"><p>Loading ShopPredict AI…</p></div>;

  return (
    <div className="shell">
      {/* Top bar — reference: logo left, black pill nav centre, icons + avatar right */}
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div className="flex items-center gap-2 font-extrabold text-lg"><Sparkles size={22} /> ShopPredict <span className="font-normal">AI</span></div>
        <nav className="pillnav" aria-label="Main navigation">
          {(['Overview', 'Predict', 'Models', 'Insights'] as Tab[]).map(t => (
            <button key={t} className={tab === t ? 'active' : ''} onClick={() => setTab(t)}>{t}</button>
          ))}
        </nav>
        <div className="flex items-center gap-2">
          <span className="iconbtn"><Search size={16} /></span>
          <span className="iconbtn"><Bell size={16} /></span>
          <span className="iconbtn"><Settings size={16} /></span>
          <span className="iconbtn" style={{ background: '#111', color: '#fff', fontWeight: 700 }}>C</span>
        </div>
      </div>

      {/* Title row — reference: back arrow + big title + Create button */}
      <div className="flex items-center justify-between mt-5 flex-wrap gap-3">
        <div className="flex items-center gap-3">
          <span className="iconbtn"><ArrowLeft size={18} /></span>
          <h1 className="text-3xl font-extrabold m-0">{tab === 'Overview' ? 'Purchase Sessions' : tab === 'Predict' ? 'Predict a Session' : tab === 'Models' ? 'Model Comparison' : 'Insights & Explainability'}</h1>
        </div>
        <div className="flex gap-2">
          <span className="iconbtn"><Filter size={16} /></span>
          <button className="btnwhite" onClick={() => setTab('Predict')}><Plus size={15} className="inline" /> New prediction</button>
        </div>
      </div>

      {error && <div className="mt-4 p-3 rounded-xl" style={{ background: '#ffe8e8' }}>{error} <button className="btnblack ml-2" onClick={() => window.location.reload()}>Retry</button></div>}

      {/* Stats row — reference layout: 3 stats + timeline on left, payout/model card on right */}
      {analytics && metrics && (
        <div className="grid md:grid-cols-3 gap-4 mt-4">
          <div className="statcard md:col-span-2">
            <div className="grid grid-cols-3 gap-4">
              <div><div className="text-xs text-gray-600">Total sessions</div><div className="text-2xl font-extrabold">{fmt(analytics.total_sessions)}</div></div>
              <div><div className="text-xs text-gray-600">Purchase sessions</div><div className="text-2xl font-extrabold">{fmt(analytics.purchase_sessions)}</div></div>
              <div><div className="text-xs text-gray-600">Purchase rate</div><div className="text-2xl font-extrabold">{pct(analytics.purchase_rate)} <span className="text-sm font-normal text-gray-500">of sessions</span></div></div>
            </div>
            <div className="mt-5">
              <div className="text-xs text-gray-600 mb-2">Purchase rate by month — May is the largest month, November purchases most often</div>
              <div className="flex gap-2 items-end">
                {monthData.slice(0, 6).map((d: any) => (
                  <div key={d.month} className="flex-1 text-center">
                    <div className="text-xs font-bold">{d.pct}%</div>
                    <div style={{ height: `${Math.max(8, d.pct * 3)}px`, background: '#d6f93c', borderRadius: 6 }} />
                    <div className="text-xs mt-1">{d.month}</div>
                  </div>
                ))}
              </div>
              <div className="flex gap-2 mt-3">
                {['Returning visitors 85.6%', 'New visitors 13.7%', 'Other 0.7%'].map(s => <span key={s} className="text-xs bg-white rounded-full px-2 py-1">{s}</span>)}
              </div>
            </div>
          </div>
          <div className="bg-white rounded-[20px] p-5 border border-gray-100">
            <div className="flex justify-between"><span className="text-xs text-gray-600">Selected model</span><span className="text-xs bg-gray-100 rounded-full px-2 py-0.5">Served by API</span></div>
            <div className="text-2xl font-extrabold mt-1">{metrics.selected_model}</div>
            <div className="text-sm text-gray-600">ROC-AUC {metrics.test_at_threshold.roc_auc.toFixed(3)} · PR-AUC {metrics.test_at_threshold.pr_auc.toFixed(3)} · Threshold {metrics.threshold}</div>
            <div className="mt-3 space-y-2">
              {models.slice(0, 3).map((m: any) => (
                <div key={m.model} className="flex justify-between text-sm p-2 rounded-lg" style={{ background: m.model === metrics.selected_model ? '#d6f93c' : '#f3f4f6' }}>
                  <span>{m.model}</span><b>{m.test.pr_auc.toFixed(3)} PR-AUC</b>
                </div>
              ))}
            </div>
            <button className="btnblack w-full mt-3" onClick={() => setTab('Predict')}>Predict now</button>
          </div>
        </div>
      )}

      {/* Filters row — reference: Active filters + dropdowns + search */}
      <div className="flex gap-2 mt-4 flex-wrap items-center">
        <span className="bg-black text-white text-xs rounded-full px-3 py-1.5">Active filters {(visitorFilter !== 'All visitors' ? 1 : 0) + (monthFilter !== 'All months' ? 1 : 0)}</span>
        <select value={visitorFilter} onChange={e => setVisitorFilter(e.target.value)} style={{ width: 'auto' }}>
          <option>All visitors</option><option>Returning_Visitor</option><option>New_Visitor</option><option>Other</option>
        </select>
        <select value={monthFilter} onChange={e => setMonthFilter(e.target.value)} style={{ width: 'auto' }}>
          <option>All months</option>{(meta?.categorical_values?.Month || []).map((m: string) => <option key={m}>{m}</option>)}
        </select>
        <span className="text-xs text-gray-500 flex items-center gap-1"><Calendar size={14} /> UCI Online Shoppers · 12,330 sessions</span>
        <span className="text-xs text-gray-500 ml-auto">{filteredNote}</span>
      </div>

      {/* Bottom dark panel — reference structure, content by tab */}
      <div className="darkpanel mt-4">
        {tab === 'Overview' && (
          <div className="grid md:grid-cols-5 gap-4">
            <div className="md:col-span-2">
              <div className="flex gap-2 mb-3">
                <button className="tabchip active">All models</button>
                <button className="tabchip" onClick={() => setTab('Models')}>Compare →</button>
              </div>
              <h3 className="font-bold mb-2">Model leaderboard</h3>
              {models.map((m: any) => (
                <div key={m.model} className={`listrow ${detailModel === m.model ? 'sel' : ''}`} onClick={() => setDetailModel(m.model)}>
                  <span className="w-9 h-9 rounded-full bg-white text-black flex items-center justify-center font-bold text-sm">{m.model[0]}</span>
                  <span className="flex-1"><b className="block text-sm">{m.model}</b><span className="text-xs text-gray-300">{m.train_time_s}s train · F1 {m.test.f1.toFixed(3)}</span></span>
                  <b className="text-sm">{m.test.pr_auc.toFixed(3)}</b>
                </div>
              ))}
            </div>
            <div className="md:col-span-3">
              <div className="flex justify-between text-xs text-gray-300 mb-2"><span>Model details</span><span>Test set · threshold {metrics?.threshold}</span></div>
              {detail && (
                <div className="detail">
                  <div className="flex justify-between flex-wrap gap-2">
                    <div><b className="text-xl">{detail.model}</b><div className="text-xs">Selected on validation PR-AUC, then F1 — test evaluated once</div></div>
                    <div className="text-right"><b>{detail.model === metrics?.selected_model ? 'Served model' : 'Comparison model'}</b><div className="text-xs">{detail.model === metrics?.selected_model ? 'Used by /api/predict' : 'Not served by /api/predict'}</div></div>
                  </div>
                  <div className="grid grid-cols-3 gap-2 mt-4">
                    <div className="mini"><div className="text-xs">Precision</div><b>{detail.test.precision.toFixed(3)}</b><div className="text-xs">of flagged purchases correct</div></div>
                    <div className="mini"><div className="text-xs">Recall</div><b>{detail.test.recall.toFixed(3)}</b><div className="text-xs">of real purchases found</div></div>
                    <div className="mini"><div className="text-xs">F1-score</div><b>{detail.test.f1.toFixed(3)}</b><div className="text-xs">balance of the two</div></div>
                  </div>
                  <div className="grid grid-cols-3 gap-2 mt-3 text-sm">
                    <div>Accuracy <b>{detail.test.accuracy.toFixed(3)}</b></div>
                    <div>ROC-AUC <b>{detail.test.roc_auc.toFixed(3)}</b></div>
                    <div>PR-AUC <b>{detail.test.pr_auc.toFixed(3)}</b></div>
                  </div>
                  <div className="flex justify-between mt-4 text-sm"><span>Confusion: TN {detail.test.tn} · FP {detail.test.fp} · FN {detail.test.fn} · TP {detail.test.tp}</span><button className="btnlime" onClick={() => setTab('Predict')}>Predict now</button></div>
                </div>
              )}
              <p className="text-xs text-gray-400 mt-3">Predictions are decision support, not guaranteed outcomes. PageValues is excluded from the served model — see Insights for the leakage comparison.</p>
            </div>
          </div>
        )}

        {tab === 'Predict' && (
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white text-black rounded-2xl p-4">
              <h3 className="font-bold mb-3">Session inputs — all fields required except PageValues (excluded, see Insights)</h3>
              <div className="grid grid-cols-2 gap-3">
                <Field label="Administrative pages"><input type="number" value={form.Administrative} onChange={e => setForm({ ...form, Administrative: +e.target.value })} /></Field>
                <Field label="Administrative duration (s)"><input type="number" value={form.Administrative_Duration} onChange={e => setForm({ ...form, Administrative_Duration: +e.target.value })} /></Field>
                <Field label="Informational pages"><input type="number" value={form.Informational} onChange={e => setForm({ ...form, Informational: +e.target.value })} /></Field>
                <Field label="Informational duration (s)"><input type="number" value={form.Informational_Duration} onChange={e => setForm({ ...form, Informational_Duration: +e.target.value })} /></Field>
                <Field label="Product-related pages"><input type="number" value={form.ProductRelated} onChange={e => setForm({ ...form, ProductRelated: +e.target.value })} /></Field>
                <Field label="Product-related duration (s)"><input type="number" value={form.ProductRelated_Duration} onChange={e => setForm({ ...form, ProductRelated_Duration: +e.target.value })} /></Field>
                <Field label="Bounce rate (0–0.2)"><input type="number" step="0.01" value={form.BounceRates} onChange={e => setForm({ ...form, BounceRates: +e.target.value })} /></Field>
                <Field label="Exit rate (0–0.2)"><input type="number" step="0.01" value={form.ExitRates} onChange={e => setForm({ ...form, ExitRates: +e.target.value })} /></Field>
                <Field label="Special day (0–1)"><input type="number" step="0.1" value={form.SpecialDay} onChange={e => setForm({ ...form, SpecialDay: +e.target.value })} /></Field>
                <Field label="Month"><select value={form.Month} onChange={e => setForm({ ...form, Month: e.target.value })}>{(meta?.categorical_values?.Month || []).map((m: string) => <option key={m}>{m}</option>)}</select></Field>
                <Field label="Operating system"><input type="number" value={form.OperatingSystems} onChange={e => setForm({ ...form, OperatingSystems: +e.target.value })} /></Field>
                <Field label="Browser"><input type="number" value={form.Browser} onChange={e => setForm({ ...form, Browser: +e.target.value })} /></Field>
                <Field label="Region"><input type="number" value={form.Region} onChange={e => setForm({ ...form, Region: +e.target.value })} /></Field>
                <Field label="Traffic type"><input type="number" value={form.TrafficType} onChange={e => setForm({ ...form, TrafficType: +e.target.value })} /></Field>
                <Field label="Visitor type"><select value={form.VisitorType} onChange={e => setForm({ ...form, VisitorType: e.target.value })}><option>Returning_Visitor</option><option>New_Visitor</option><option>Other</option></select></Field>
                <Field label="Weekend"><select value={form.Weekend ? 'true' : 'false'} onChange={e => setForm({ ...form, Weekend: e.target.value === 'true' })}><option value="false">No</option><option value="true">Yes</option></select></Field>
              </div>
              <button className="btnblack w-full mt-4" onClick={doPredict} disabled={predicting}>{predicting ? 'Predicting…' : 'Predict purchase intention'}</button>
            </div>
            <div>
              <div className="detail">
                <h3 className="font-bold">Prediction result</h3>
                {!result && <p className="text-sm">Fill the form and predict. The probability bar and threshold below update from the live API — nothing here is hardcoded.</p>}
                {result?.error && <p className="text-sm" style={{ color: '#7a0000' }}>{result.error}</p>}
                {result && !result.error && (
                  <div>
                    <div className="text-2xl font-extrabold">{result.label}</div>
                    <div className="text-sm">Probability {pct(result.purchase_probability)} · Threshold {result.threshold} · {result.selected_model}</div>
                    <div className="h-4 bg-white/40 rounded-full mt-3 overflow-hidden"><div style={{ width: `${result.purchase_probability * 100}%`, height: '100%', background: '#d6f93c' }} /></div>
                    <div className="text-xs mt-1">Threshold line: {pct(result.threshold)} — above it counts as “purchase likely”.</div>
                    {result.explanation?.length ? <ul className="text-sm mt-3 space-y-1">{result.explanation.map((x: any, i: number) => <li key={i}>• <b>{x.feature}</b>: {x.note}</li>)}</ul> : <p className="text-sm mt-3">No single input stood out in this session — see Insights for the global feature ranking.</p>}
                    <p className="text-xs mt-3">{result.note}</p>
                  </div>
                )}
              </div>
              <div className="mt-3 bg-white/10 rounded-2xl p-4">
                <div className="flex justify-between"><b>Prediction history (this browser only)</b><button className="text-xs flex items-center gap-1" onClick={() => { setHistory([]); localStorage.removeItem('shoppredict_history'); }}><Trash2 size={13} /> Clear</button></div>
                {history.length === 0 ? <p className="text-xs text-gray-300 mt-2">No predictions yet. History is not sent anywhere and is not verified purchases.</p> : history.slice(0, 6).map((h: any, i: number) => <div key={i} className="text-sm flex justify-between py-1 border-b border-white/10"><span>{h.label} · {pct(h.purchase_probability)}</span><span className="text-xs text-gray-300">{h.at}</span></div>)}
              </div>
            </div>
          </div>
        )}

        {tab === 'Models' && metrics && (
          <div>
            <div className="overflow-x-auto bg-white text-black rounded-2xl p-3">
              <table className="w-full text-sm">
                <thead><tr className="text-left border-b"><th className="py-2">Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th><th>PR-AUC</th><th>Train (s)</th></tr></thead>
                <tbody>{models.map((m: any) => <tr key={m.model} className="border-b" style={{ background: m.model === metrics.selected_model ? '#f3ffd6' : 'transparent' }}><td className="py-2 font-bold">{m.model}{m.model === metrics.selected_model ? ' ★ selected' : ''}</td><td>{m.test.accuracy.toFixed(3)}</td><td>{m.test.precision.toFixed(3)}</td><td>{m.test.recall.toFixed(3)}</td><td>{m.test.f1.toFixed(3)}</td><td>{m.test.roc_auc.toFixed(3)}</td><td>{m.test.pr_auc.toFixed(3)}</td><td>{m.train_time_s}</td></tr>)}</tbody>
              </table>
              <p className="text-xs text-gray-500 mt-2">All models evaluated once on the same untouched test split, at threshold 0.5 in this table. The served model uses its validation-tuned threshold on the Predict tab.</p>
            </div>
            <div className="grid md:grid-cols-2 gap-4 mt-4">
              <div className="bg-white text-black rounded-2xl p-4"><h4 className="font-bold">ROC curve — selected model (test)</h4><div style={{ height: 220 }}><ResponsiveContainer><LineChart data={metrics.curves.roc}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="x" label={{ value: 'False positive rate', position: 'insideBottom', offset: -2 }} /><YAxis /><Tooltip /><Line type="monotone" dataKey="y" stroke="#111" dot={false} /></LineChart></ResponsiveContainer></div></div>
              <div className="bg-white text-black rounded-2xl p-4"><h4 className="font-bold">Precision–Recall curve — selected model (test)</h4><div style={{ height: 220 }}><ResponsiveContainer><LineChart data={metrics.curves.pr}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="x" label={{ value: 'Recall', position: 'insideBottom', offset: -2 }} /><YAxis /><Tooltip /><Line type="monotone" dataKey="y" stroke="#5b21b6" dot={false} /></LineChart></ResponsiveContainer></div></div>
            </div>
          </div>
        )}

        {tab === 'Insights' && analytics && metrics && (
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white text-black rounded-2xl p-4">
              <h4 className="font-bold">Global feature importance (permutation, test subsample, ROC-AUC)</h4>
              <p className="text-xs text-gray-500">How much the score drops when a feature is shuffled. Associations only — not proven causes.</p>
              <div style={{ height: 300 }}><ResponsiveContainer><BarChart data={importance} layout="vertical"><CartesianGrid strokeDasharray="3 3" /><XAxis type="number" /><YAxis type="category" dataKey="feature" width={150} /><Tooltip /><Bar dataKey="importance" fill="#111">{importance.map((_: any, i: number) => <Cell key={i} fill={i === 0 ? '#d6f93c' : '#111'} />)}</Bar></BarChart></ResponsiveContainer></div>
            </div>
            <div className="bg-white text-black rounded-2xl p-4">
              <h4 className="font-bold">Leakage check — PageValues</h4>
              <p className="text-sm">XGBoost <b>with</b> PageValues, validation PR-AUC: <b>{metrics.leakage_comparison.xgboost_with_pagevalues_validation?.pr_auc?.toFixed(3)}</b><br />XGBoost <b>without</b> PageValues, validation PR-AUC: <b>{metrics.leakage_comparison.xgboost_without_pagevalues_validation?.pr_auc?.toFixed(3)}</b></p>
              <p className="text-sm mt-2">PageValues averages {analytics.pagevalues_mean_purchase.toFixed(1)} for purchases vs {analytics.pagevalues_mean_no_purchase.toFixed(1)} for non-purchases — it tracks completed-transaction value, so it is excluded from the served model even if it lifts the score. Prediction moment: during browsing, before checkout completes.</p>
              <h4 className="font-bold mt-4">Purchase rate by visitor type</h4>
              {visitorData.map((v: any) => <div key={v.name} className="flex justify-between text-sm py-1 border-b"><span>{v.name}</span><b>{pct(v.rate)}</b></div>)}
              <h4 className="font-bold mt-4">Imbalance</h4>
              <p className="text-sm">Only 15.5% of sessions purchase, so accuracy alone misleads. Class-weighted models were used; a no-weight vs balanced Logistic Regression check is in the report. SMOTE was not used — resampling was not needed once class weights closed the gap, and it would risk duplicating minority sessions inside folds.</p>
            </div>
          </div>
        )}
      </div>
      <p className="text-xs text-gray-600 mt-3 text-center">ShopPredict AI · UCI Online Shoppers Purchasing Intention Dataset (12,330 sessions) · Reference layout: Salesforce Invoices screenshot, content adapted to purchase prediction · Predictions are decision support only.</p>
    </div>
  );
}
