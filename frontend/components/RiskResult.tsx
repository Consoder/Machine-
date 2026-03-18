'use client'
import ShapChart from './ShapChart'

interface Feature {
  feature: string
  shap_value: number
  direction: string
}
interface Result {
  risk_score: number
  risk_label: string
  risk_percent: string
  prediction: number
  threshold: number
  top_features: Feature[]
}

export default function RiskResult({ result }: { result: Result }) {
  const colorMap: Record<string, string> = {
    HIGH:   'bg-red-50 border-red-200 text-red-700',
    MEDIUM: 'bg-yellow-50 border-yellow-200 text-yellow-700',
    LOW:    'bg-green-50 border-green-200 text-green-700',
  }
  const barColorMap: Record<string, string> = {
    HIGH: 'bg-red-500', MEDIUM: 'bg-yellow-400', LOW: 'bg-green-500',
  }
  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6 space-y-6">
      <div className={`rounded-xl border px-5 py-4 ${colorMap[result.risk_label]}`}>
        <div className="flex justify-between items-center">
          <div>
            <p className="text-xs font-semibold uppercase tracking-widest opacity-70">Readmission Risk</p>
            <p className="text-4xl font-bold mt-1">{result.risk_percent}</p>
          </div>
          <span className={`text-lg font-bold px-4 py-2 rounded-lg border ${colorMap[result.risk_label]}`}>
            {result.risk_label}
          </span>
        </div>
        <div className="mt-4 bg-white bg-opacity-50 rounded-full h-3 overflow-hidden">
          <div
            className={`h-3 rounded-full transition-all duration-700 ${barColorMap[result.risk_label]}`}
            style={{ width: result.risk_percent }}
          />
        </div>
      </div>
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-gray-50 rounded-xl p-4 text-center">
          <p className="text-xs text-gray-400 uppercase tracking-wide">Risk Score</p>
          <p className="text-2xl font-semibold text-gray-800 mt-1">{result.risk_score.toFixed(3)}</p>
        </div>
        <div className="bg-gray-50 rounded-xl p-4 text-center">
          <p className="text-xs text-gray-400 uppercase tracking-wide">Threshold</p>
          <p className="text-2xl font-semibold text-gray-800 mt-1">{result.threshold.toFixed(3)}</p>
        </div>
        <div className="bg-gray-50 rounded-xl p-4 text-center">
          <p className="text-xs text-gray-400 uppercase tracking-wide">Prediction</p>
          <p className={`text-2xl font-semibold mt-1 ${result.prediction === 1 ? 'text-red-600' : 'text-green-600'}`}>
            {result.prediction === 1 ? 'At Risk' : 'Safe'}
          </p>
        </div>
      </div>
      {result.top_features.length > 0 && (
        <div className="border-t border-gray-100 pt-6">
          <ShapChart features={result.top_features} />
        </div>
      )}
      <p className="text-xs text-gray-400 text-center">Model: XGBoost v1 · Explainability: SHAP TreeExplainer</p>
    </div>
  )
}
