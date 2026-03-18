'use client'
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
          <Tooltip formatter={(value: any) => [Number(value).toFixed(3), 'SHAP Value']} />
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
