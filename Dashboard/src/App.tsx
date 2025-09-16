import React, { useState, useEffect } from 'react';
import { 
  TrendingUp, 
  TrendingDown, 
  AlertTriangle, 
  CheckCircle, 
  Info, 
  Database,
  Target,
  BarChart3,
  PieChart as PieChartIcon,
  Users,
  GraduationCap,
  Briefcase
} from 'lucide-react';

// Mock data based on the provided JSON files
const dashboardData = {
  executiveSummary: {
    projectName: "DataPros Adult Income Classification",
    analysisDate: "2024-09-15",
    overallStatus: "MODERATE_PERFORMANCE",
    totalRecords: 2000,
    modelAuc: 0.531,
    modelAccuracy: 0.534,
    classBalance: { high: 48.7, low: 51.2 }
  },
  kpiMetrics: [
    { title: "Total Records", value: "2,000", trend: "stable", icon: Database, status: "excellent" },
    { title: "Model AUC", value: "0.531", trend: "needs_improvement", icon: TrendingUp, status: "moderate" },
    { title: "Accuracy", value: "53.4%", trend: "moderate", icon: Target, status: "moderate" },
    { title: "Class Balance", value: "48.7% / 51.2%", trend: "excellent", icon: BarChart3, status: "excellent" }
  ],
  charts: {
    incomeDistribution: [
      { label: ">50K", value: 974, percentage: 48.7, color: "#10b981" },
      { label: "≤50K", value: 1025, percentage: 51.2, color: "#ef4444" }
    ],
    educationIncome: [
      { category: "Assoc", value: 52.2, count: 163, total: 312 },
      { category: "Masters", value: 52.1, count: 174, total: 334 },
      { category: "Some-college", value: 51.6, count: 189, total: 366 },
      { category: "HS-grad", value: 47.5, count: 153, total: 322 },
      { category: "11th", value: 45.4, count: 159, total: 350 },
      { category: "Bachelors", value: 43.2, count: 136, total: 315 }
    ],
    genderIncome: [
      { category: "Female", value: 49.0, count: 502, total: 1025 },
      { category: "Male", value: 48.4, count: 472, total: 975 }
    ],
    workclassIncome: [
      { category: "Self-emp", value: 50.8, count: 254, total: 500 },
      { category: "Private", value: 48.9, count: 244, total: 499 },
      { category: "Gov", value: 48.7, count: 243, total: 499 }
    ]
  },
  modelPerformance: {
    auc: 0.531,
    accuracy: 0.534,
    precision: 0.533,
    recall: 0.534,
    f1Score: 0.533
  },
  confusionMatrix: [
    { actual: "≤50K", predicted: { low: 95, high: 10 } },
    { actual: ">50K", predicted: { low: 8, high: 87 } }
  ],
  alerts: [
    {
      type: "critical",
      title: "Low Model Confidence",
      message: "Average model confidence is 49.1%, indicating uncertain predictions.",
      severity: "high"
    },
    {
      type: "warning", 
      title: "Moderate AUC",
      message: "AUC of 0.531 indicates moderate discrimination capability.",
      severity: "medium"
    },
    {
      type: "success",
      title: "Balanced Classes",
      message: "Classes are well balanced (48.7% vs 51.2%)",
      severity: "low"
    }
  ],
  findings: [
    {
      category: "Data Quality",
      title: "Balanced Class Distribution",
      description: "Classes are well balanced: 48.7% >50K vs 51.2% ≤50K",
      impact: "positive"
    },
    {
      category: "Model Performance",
      title: "Moderate Model Performance", 
      description: "AUC of 0.531 indicates moderate discrimination capability",
      impact: "warning"
    },
    {
      category: "Feature Analysis",
      title: "Education Impact on Income",
      description: "'Assoc' has the highest rate of >50K income: 52.2%",
      impact: "insight"
    }
  ]
};

const StatusBadge = ({ status }: { status: string }) => {
  const getStatusColor = () => {
    switch (status) {
      case 'excellent': return 'bg-green-100 text-green-800 border-green-200';
      case 'good': return 'bg-blue-100 text-blue-800 border-blue-200';
      case 'moderate': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'poor': return 'bg-red-100 text-red-800 border-red-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  return (
    <span className={`px-2 py-1 rounded-full text-xs font-medium border ${getStatusColor()}`}>
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  );
};

const KPICard = ({ title, value, trend, icon: Icon, status }: any) => {
  const getTrendIcon = () => {
    switch (trend) {
      case 'excellent':
      case 'good':
        return <TrendingUp className="w-4 h-4 text-green-500" />;
      case 'needs_improvement':
      case 'poor':
        return <TrendingDown className="w-4 h-4 text-red-500" />;
      default:
        return <div className="w-4 h-4 bg-gray-300 rounded-full" />;
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow">
      <div className="flex items-center justify-between mb-4">
        <Icon className="w-8 h-8 text-blue-600" />
        {getTrendIcon()}
      </div>
      <div className="space-y-2">
        <p className="text-sm font-medium text-gray-600">{title}</p>
        <p className="text-3xl font-bold text-gray-900">{value}</p>
        <StatusBadge status={status} />
      </div>
    </div>
  );
};

const PieChart = ({ data }: { data: any[] }) => {
  const total = data.reduce((sum, item) => sum + item.value, 0);
  let cumulativePercentage = 0;

  return (
    <div className="flex items-center justify-center space-x-8">
      <div className="relative w-48 h-48">
        <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
          {data.map((item, index) => {
            const percentage = (item.value / total) * 100;
            const strokeDasharray = `${percentage} ${100 - percentage}`;
            const strokeDashoffset = -cumulativePercentage;
            cumulativePercentage += percentage;

            return (
              <circle
                key={index}
                cx="50"
                cy="50"
                r="40"
                fill="transparent"
                stroke={item.color}
                strokeWidth="12"
                strokeDasharray={strokeDasharray}
                strokeDashoffset={strokeDashoffset}
                className="transition-all duration-300 hover:stroke-width-[14]"
              />
            );
          })}
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900">{total.toLocaleString()}</div>
            <div className="text-sm text-gray-600">Total Records</div>
          </div>
        </div>
      </div>
      <div className="space-y-3">
        {data.map((item, index) => (
          <div key={index} className="flex items-center space-x-3">
            <div
              className="w-4 h-4 rounded-full"
              style={{ backgroundColor: item.color }}
            />
            <div className="flex-1">
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium text-gray-900">{item.label}</span>
                <span className="text-sm text-gray-600">{item.percentage}%</span>
              </div>
              <div className="text-xs text-gray-500">{item.value.toLocaleString()} records</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

const BarChart = ({ data, title, yAxisLabel }: { data: any[], title: string, yAxisLabel: string }) => {
  const maxValue = Math.max(...data.map(item => item.value));
  
  return (
    <div className="space-y-4">
      <h4 className="text-sm font-medium text-gray-600">{yAxisLabel}</h4>
      <div className="space-y-3">
        {data.map((item, index) => (
          <div key={index} className="flex items-center space-x-4">
            <div className="w-20 text-sm text-gray-600 text-right">{item.category}</div>
            <div className="flex-1 flex items-center space-x-2">
              <div className="flex-1 bg-gray-100 rounded-full h-6 relative overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-blue-500 to-blue-600 rounded-full transition-all duration-500 hover:from-blue-600 hover:to-blue-700"
                  style={{ width: `${(item.value / maxValue) * 100}%` }}
                />
                <div className="absolute inset-0 flex items-center justify-center text-xs font-medium text-white">
                  {item.value.toFixed(1)}%
                </div>
              </div>
              <div className="text-xs text-gray-500 min-w-[4rem] text-right">
                {item.count}/{item.total}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

const PerformanceRadar = ({ data }: { data: any }) => {
  const metrics = [
    { name: 'AUC', value: data.auc },
    { name: 'Accuracy', value: data.accuracy },
    { name: 'Precision', value: data.precision },
    { name: 'Recall', value: data.recall },
    { name: 'F1-Score', value: data.f1Score }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
      {metrics.map((metric, index) => (
        <div key={index} className="text-center space-y-2">
          <div className="relative w-20 h-20 mx-auto">
            <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
              <circle
                cx="50"
                cy="50"
                r="40"
                fill="transparent"
                stroke="#e5e7eb"
                strokeWidth="8"
              />
              <circle
                cx="50"
                cy="50"
                r="40"
                fill="transparent"
                stroke="#3b82f6"
                strokeWidth="8"
                strokeDasharray={`${metric.value * 100 * 2.51} 251.2`}
                strokeLinecap="round"
                className="transition-all duration-500"
              />
            </svg>
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="text-sm font-bold text-gray-900">
                {(metric.value * 100).toFixed(1)}%
              </span>
            </div>
          </div>
          <div className="text-sm font-medium text-gray-600">{metric.name}</div>
        </div>
      ))}
    </div>
  );
};

const AlertCard = ({ alert }: { alert: any }) => {
  const getAlertStyles = () => {
    switch (alert.type) {
      case 'critical':
        return 'border-red-200 bg-red-50 text-red-800';
      case 'warning':
        return 'border-yellow-200 bg-yellow-50 text-yellow-800';
      case 'success':
        return 'border-green-200 bg-green-50 text-green-800';
      default:
        return 'border-blue-200 bg-blue-50 text-blue-800';
    }
  };

  const getIcon = () => {
    switch (alert.type) {
      case 'critical':
        return <AlertTriangle className="w-5 h-5 text-red-500" />;
      case 'warning':
        return <AlertTriangle className="w-5 h-5 text-yellow-500" />;
      case 'success':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      default:
        return <Info className="w-5 h-5 text-blue-500" />;
    }
  };

  return (
    <div className={`border rounded-lg p-4 ${getAlertStyles()}`}>
      <div className="flex items-start space-x-3">
        {getIcon()}
        <div className="flex-1 min-w-0">
          <h4 className="text-sm font-medium mb-1">{alert.title}</h4>
          <p className="text-sm opacity-90">{alert.message}</p>
        </div>
      </div>
    </div>
  );
};

const ConfusionMatrix = ({ data }: { data: any[] }) => {
  return (
    <div className="overflow-hidden rounded-lg border border-gray-200">
      <table className="min-w-full bg-white">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Actual \ Predicted</th>
            <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase">≤50K</th>
            <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase">50K</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-200">
          {data.map((row, index) => (
            <tr key={index} className="hover:bg-gray-50">
              <td className="px-4 py-3 text-sm font-medium text-gray-900">{row.actual}</td>
              <td className="px-4 py-3 text-center text-sm text-gray-600">{row.predicted.low}</td>
              <td className="px-4 py-3 text-center text-sm text-gray-600">{row.predicted.high}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

function App() {
  const [activeTab, setActiveTab] = useState('overview');

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <BarChart3 className="w-8 h-8 text-blue-600" />
                <div>
                  <h1 className="text-xl font-bold text-gray-900">DataPros ML Dashboard</h1>
                  <p className="text-sm text-gray-500">Adult Income Classification</p>
                </div>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <StatusBadge status="moderate" />
              <div className="text-sm text-gray-600">
                Last Updated: {dashboardData.executiveSummary.analysisDate}
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {[
              { id: 'overview', label: 'Overview', icon: BarChart3 },
              { id: 'performance', label: 'Model Performance', icon: Target },
              { id: 'insights', label: 'Data Insights', icon: PieChartIcon },
              { id: 'alerts', label: 'Alerts & Findings', icon: AlertTriangle }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 px-3 py-4 border-b-2 text-sm font-medium transition-colors ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <tab.icon className="w-4 h-4" />
                <span>{tab.label}</span>
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'overview' && (
          <div className="space-y-8">
            {/* KPI Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {dashboardData.kpiMetrics.map((kpi, index) => (
                <KPICard key={index} {...kpi} />
              ))}
            </div>

            {/* Executive Summary */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <h2 className="text-xl font-bold text-gray-900 mb-6">Executive Summary</h2>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div className="space-y-6">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Project Overview</h3>
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Project:</span>
                        <span className="font-medium">{dashboardData.executiveSummary.projectName}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Analysis Date:</span>
                        <span className="font-medium">{dashboardData.executiveSummary.analysisDate}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Total Records:</span>
                        <span className="font-medium">{dashboardData.executiveSummary.totalRecords.toLocaleString()}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Overall Status:</span>
                        <StatusBadge status="moderate" />
                      </div>
                    </div>
                  </div>
                </div>
                <div className="space-y-6">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Key Metrics</h3>
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Model AUC:</span>
                        <span className="font-medium">{dashboardData.executiveSummary.modelAuc.toFixed(3)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Accuracy:</span>
                        <span className="font-medium">{(dashboardData.executiveSummary.modelAccuracy * 100).toFixed(1)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Class Balance:</span>
                        <span className="font-medium">{dashboardData.executiveSummary.classBalance.high}% / {dashboardData.executiveSummary.classBalance.low}%</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Income Distribution */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-6">Income Distribution</h3>
              <PieChart data={dashboardData.charts.incomeDistribution} />
            </div>
          </div>
        )}

        {activeTab === 'performance' && (
          <div className="space-y-8">
            {/* Model Performance Metrics */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <h2 className="text-xl font-bold text-gray-900 mb-6">Model Performance Metrics</h2>
              <PerformanceRadar data={dashboardData.modelPerformance} />
            </div>

            {/* Confusion Matrix */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-6">Confusion Matrix</h3>
              <ConfusionMatrix data={dashboardData.confusionMatrix} />
            </div>
          </div>
        )}

        {activeTab === 'insights' && (
          <div className="space-y-8">
            {/* Education Analysis */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <div className="flex items-center space-x-2 mb-6">
                <GraduationCap className="w-6 h-6 text-blue-600" />
                <h3 className="text-lg font-semibold text-gray-900">Education Impact on Income</h3>
              </div>
              <BarChart 
                data={dashboardData.charts.educationIncome} 
                title="Education vs High Income Rate"
                yAxisLabel="Percentage of High Income (>50K)"
              />
            </div>

            {/* Gender Analysis */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                <div className="flex items-center space-x-2 mb-6">
                  <Users className="w-6 h-6 text-blue-600" />
                  <h3 className="text-lg font-semibold text-gray-900">Gender Analysis</h3>
                </div>
                <BarChart 
                  data={dashboardData.charts.genderIncome} 
                  title="Gender vs High Income Rate"
                  yAxisLabel="Percentage of High Income (>50K)"
                />
              </div>

              <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
                <div className="flex items-center space-x-2 mb-6">
                  <Briefcase className="w-6 h-6 text-blue-600" />
                  <h3 className="text-lg font-semibold text-gray-900">Work Class Analysis</h3>
                </div>
                <BarChart 
                  data={dashboardData.charts.workclassIncome} 
                  title="Work Class vs High Income Rate"
                  yAxisLabel="Percentage of High Income (>50K)"
                />
              </div>
            </div>
          </div>
        )}

        {activeTab === 'alerts' && (
          <div className="space-y-8">
            {/* Alerts */}
            <div className="space-y-4">
              <h2 className="text-xl font-bold text-gray-900">Alerts & Notifications</h2>
              {dashboardData.alerts.map((alert, index) => (
                <AlertCard key={index} alert={alert} />
              ))}
            </div>

            {/* Key Findings */}
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-6">Key Findings</h3>
              <div className="space-y-6">
                {dashboardData.findings.map((finding, index) => (
                  <div key={index} className="border-l-4 border-blue-500 pl-4 py-2">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-semibold text-gray-900">{finding.title}</h4>
                      <span className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded-full">
                        {finding.category}
                      </span>
                    </div>
                    <p className="text-gray-600 text-sm">{finding.description}</p>
                    <div className="mt-2">
                      <StatusBadge 
                        status={finding.impact === 'positive' ? 'excellent' : finding.impact === 'warning' ? 'moderate' : 'good'} 
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;