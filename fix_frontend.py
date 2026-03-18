import pathlib

pathlib.Path("frontend/components").mkdir(parents=True, exist_ok=True)

pathlib.Path("frontend/app/page.tsx").write_text(
"""'use client'
import { useState } from 'react'
import PatientForm from '../components/PatientForm'
import RiskResult from '../components/RiskResult'

export default function Home() {
  const [result, setResult] = useState<any>(null)
  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      <div className="bg-white border-b border-gray-100 shadow-sm">
        <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold text-gray-900">Hospital Readmission Predictor</h1>
            <p className="text-sm text-gray-400">XGBoost + SHAP · 30-day readmission risk</p>
          </div>
          <span className="text-xs bg-green-100 text-green-700 px-3 py-1 rounded-full font-medium">
            Model v1.0 Live
          </span>
        </div>
      </div>
      <div className="max-w-6xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <PatientForm onResult={setResult} />
          <div>
            {result ? (
              <RiskResult result={result} />
            ) : (
              <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8 h-full flex flex-col items-center justify-center text-center">
                <div className="text-6xl mb-4">🏥</div>
                <h3 className="text-lg font-semibold text-gray-700 mb-2">No prediction yet</h3>
                <p className="text-sm text-gray-400 max-w-xs">
                  Fill in patient details and click Predict to see the readmission risk score and SHAP explanation.
                </p>
              </div>
            )}
          </div>
        </div>
        <p className="text-center text-xs text-gray-400 mt-8">
          Built with XGBoost · FastAPI · Next.js · SHAP
        </p>
      </div>
    </main>
  )
}
""", encoding="utf-8")
print("page.tsx OK")

pathlib.Path("frontend/components/ShapChart.tsx").write_text(
"""'use client'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell, ResponsiveContainer } from 'recharts'

interface Feature {
  feature: string
  shap_value: number
  direction: string
}

export default function ShapChart({ features }: { features: Feature[] }) {
  const data = features.map(f => ({
    name: f.feature.replace(/_/g, ' '),
    value: parseFloat(f.shap_value.toFixed(3)),
    direction: f.direction,
  }))
  return (
    <div className="w-full">
      <h3 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-4">
        Top Factors Driving This Prediction
      </h3>
      <ResponsiveContainer width="100%" height={320}>
        <BarChart data={data} layout="vertical" margin={{ top: 0, right: 30, left: 140, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" horizontal={false} />
          <XAxis type="number" tick={{ fontSize: 12 }} />
          <YAxis type="category" dataKey="name" tick={{ fontSize: 12 }} width={130} />
          <Tooltip formatter={(value: number) => [value.toFixed(3), 'SHAP Value']} />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {data.map((entry, index) => (
              <Cell key={index} fill={entry.direction === 'increases_risk' ? '#ef4444' : '#22c55e'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="flex gap-6 mt-3 text-sm text-gray-500">
        <span className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-sm bg-red-500 inline-block"></span>Increases risk
        </span>
        <span className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-sm bg-green-500 inline-block"></span>Decreases risk
        </span>
      </div>
    </div>
  )
}
""", encoding="utf-8")
print("ShapChart.tsx OK")

pathlib.Path("frontend/components/RiskResult.tsx").write_text(
"""'use client'
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
""", encoding="utf-8")
print("RiskResult.tsx OK")

pathlib.Path("frontend/components/PatientForm.tsx").write_text(
"""'use client'
import { useState } from 'react'

const defaultValues = {
  time_in_hospital: 5, num_lab_procedures: 41, num_procedures: 1,
  num_medications: 12, number_outpatient: 0, number_emergency: 1,
  number_inpatient: 2, number_diagnoses: 7, age: 65,
  gender: 'Female', race: 'Caucasian', admission_type_id: 1,
  discharge_disposition_id: 1, admission_source_id: 7,
  medical_specialty: 'InternalMedicine', max_glu_serum: 'None',
  A1Cresult: 'None', change: 'Ch', diabetesMed: 'Yes',
  metformin: 'Steady', insulin: 'Up',
  diag_1_group: 'circulatory', diag_2_group: 'diabetes', diag_3_group: 'respiratory',
}

const numFields = [
  { key: 'time_in_hospital',   label: 'Days in Hospital',       min: 1,  max: 14  },
  { key: 'num_lab_procedures', label: 'Lab Procedures',         min: 0,  max: 132 },
  { key: 'num_procedures',     label: 'Procedures',             min: 0,  max: 6   },
  { key: 'num_medications',    label: 'Medications',            min: 0,  max: 81  },
  { key: 'number_outpatient',  label: 'Prior Outpatient Visits',min: 0           },
  { key: 'number_emergency',   label: 'Prior Emergency Visits', min: 0           },
  { key: 'number_inpatient',   label: 'Prior Inpatient Visits', min: 0           },
  { key: 'number_diagnoses',   label: 'Number of Diagnoses',    min: 0,  max: 16  },
  { key: 'age',                label: 'Age',                    min: 5,  max: 95  },
]

const selFields = [
  { key: 'gender',          label: 'Gender',             options: ['Female','Male'] },
  { key: 'race',            label: 'Race',               options: ['Caucasian','AfricanAmerican','Hispanic','Asian','Other'] },
  { key: 'medical_specialty',label:'Specialty',          options: ['InternalMedicine','Cardiology','Surgery','Family/GeneralPractice','Other'] },
  { key: 'max_glu_serum',   label: 'Glucose Serum',      options: ['None','Norm','>200','>300'] },
  { key: 'A1Cresult',       label: 'HbA1c Result',       options: ['None','Norm','>7','>8'] },
  { key: 'change',          label: 'Med Change',         options: ['Ch','No'] },
  { key: 'diabetesMed',     label: 'Diabetes Meds',      options: ['Yes','No'] },
  { key: 'insulin',         label: 'Insulin',            options: ['No','Steady','Down','Up'] },
  { key: 'diag_1_group',    label: 'Primary Diagnosis',  options: ['circulatory','respiratory','digestive','diabetes','injury','musculoskeletal','other'] },
  { key: 'diag_2_group',    label: 'Secondary Diagnosis',options: ['circulatory','respiratory','digestive','diabetes','injury','other'] },
]

export default function PatientForm({ onResult }: { onResult: (r: any) => void }) {
  const [form, setForm]       = useState<any>(defaultValues)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState('')

  const handleSubmit = async () => {
    setLoading(true)
    setError('')
    try {
      const res = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      })
      if (!res.ok) throw new Error(await res.text())
      onResult(await res.json())
    } catch (e: any) {
      setError('API error: ' + e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
      <h2 className="text-lg font-semibold text-gray-800 mb-1">Patient Details</h2>
      <p className="text-sm text-gray-400 mb-6">Fill in clinical information to predict readmission risk</p>
      <div className="grid grid-cols-2 gap-4 mb-4">
        {numFields.map(f => (
          <div key={f.key}>
            <label className="block text-xs font-medium text-gray-500 mb-1">{f.label}</label>
            <input type="number" min={f.min} max={f.max} value={form[f.key]}
              onChange={e => setForm({ ...form, [f.key]: Number(e.target.value) })}
              className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-4 mb-6">
        {selFields.map(s => (
          <div key={s.key}>
            <label className="block text-xs font-medium text-gray-500 mb-1">{s.label}</label>
            <select value={form[s.key]}
              onChange={e => setForm({ ...form, [s.key]: e.target.value })}
              className="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
            >
              {s.options.map(opt => <option key={opt} value={opt}>{opt}</option>)}
            </select>
          </div>
        ))}
      </div>
      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-600">{error}</div>
      )}
      <button onClick={handleSubmit} disabled={loading}
        className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-blue-300 text-white font-semibold py-3 rounded-xl transition-colors duration-200">
        {loading ? 'Predicting...' : 'Predict Readmission Risk'}
      </button>
    </div>
  )
}
""", encoding="utf-8")
print("PatientForm.tsx OK")

print()
print("ALL FRONTEND FILES FIXED")
print("Now run: cd frontend && npm run dev")