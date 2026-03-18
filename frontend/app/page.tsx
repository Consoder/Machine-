'use client'
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
