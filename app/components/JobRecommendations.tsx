import React, {useEffect, useState} from 'react'

type Rec = { title: string; score: number; reason?: string };

const JobRecommendations: React.FC<{ feedback: Feedback; pdfText?: string | null }> = ({ feedback, pdfText }) => {
  const [recs, setRecs] = useState<Rec[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchRecs = async () => {
      setLoading(true);
      setError(null);

      // Prefer using original PDF text (provided by user) if available.
      let text = '';
      if (pdfText && pdfText.trim().length > 50) {
        text = pdfText;
      } else {
        // Build a simple text summary from the feedback to send to the recommender
        const parts: string[] = [];
        parts.push(`overallScore:${feedback.overallScore}`);

        const collectTips = (section: any, name: string) => {
          if (!section) return;
          // include both short tip and explanation if available
          const tipsArr = (section.tips || []).map((t: any) => {
            const tip = t.tip || '';
            const explanation = t.explanation || '';
            return [tip, explanation].filter(Boolean).join('. ');
          });
          const tips = tipsArr.join('. ');
          if (tips) parts.push(`${name}:${tips}`);
        }

        collectTips(feedback.ATS, 'ats');
        collectTips(feedback.toneAndStyle, 'tone');
        collectTips(feedback.content, 'content');
        collectTips(feedback.structure, 'structure');
        collectTips(feedback.skills, 'skills');

        text = parts.join('. ');
      }

        try {
          // Try embeddings-based recommender first (port 5001). If it's not available, fall back to keyword server (5000).
          let res = await fetch('http://127.0.0.1:5001/recommend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, n: 5 }),
          });

          if (!res.ok) {
            // fallback
            res = await fetch('http://127.0.0.1:5000/recommend', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ text, n: 5 }),
            });
          }

          if (!res.ok) throw new Error(`Server returned ${res.status}`);
          const json = await res.json();
          setRecs(json.recommendations || null);
        } catch (err: any) {
          setError('Recommendation service not available. Run the local recommender server(s) if you want live suggestions.');
          setRecs(null);
        } finally {
          setLoading(false);
        }
    }

    fetchRecs();
  }, [feedback, pdfText]);
  // include pdfText as dependency so recommendations refresh when text is available
  // (React's linting prefers listing, but we keep simple: rebuild when pdfText changes)

  return (
    <div className="bg-white rounded-2xl shadow-md w-full p-6">
      <h3 className="text-xl font-bold mb-3">Job Recommendations</h3>
      {loading && <p className="text-gray-600">Loading recommendations…</p>}
      {error && <p className="text-sm text-amber-600">{error}</p>}

      {recs && recs.length > 0 && (
        <ul className="space-y-3">
          {recs.map((r, i) => (
            <li key={i} className="flex items-center justify-between">
              <div>
                <p className="font-semibold">{r.title}</p>
                {r.reason && <p className="text-sm text-gray-600">{r.reason}</p>}
              </div>
              <div className="text-right">
                <p className="text-lg font-bold">{r.score}%</p>
                <p className="text-xs text-gray-500">match</p>
              </div>
            </li>
          ))}
        </ul>
      )}

      {!loading && !recs && !error && (
        <p className="text-gray-600">No recommendations available.</p>
      )}
    </div>
  )
}

export default JobRecommendations
