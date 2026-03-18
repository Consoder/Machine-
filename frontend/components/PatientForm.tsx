'use client'
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
