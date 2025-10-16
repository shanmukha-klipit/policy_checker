import React, { useState } from 'react';
import { Upload, FileText, CheckCircle, XCircle, Loader2 } from 'lucide-react';

const ComplianceChecker = () => {
  const [activeTab, setActiveTab] = useState('policy');
  const [policyFile, setPolicyFile] = useState(null);
  const [billFile, setBillFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [policyStatus, setPolicyStatus] = useState(null);

  // ✅ New metadata fields
  const [companyName, setCompanyName] = useState('');
  const [policyName, setPolicyName] = useState('');
  const [effectiveFrom, setEffectiveFrom] = useState('');
  const [effectiveTo, setEffectiveTo] = useState('');
  const [description, setDescription] = useState('');

  const API_BASE = 'https://policy-checker-mvf1.onrender.com';

  const handlePolicyUpload = async () => {
    if (!policyFile || !companyName.trim() || !policyName.trim()) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', policyFile);
    formData.append('company', companyName.trim());
    formData.append('policy_name', policyName.trim());
    formData.append('effective_from', effectiveFrom || '');
    formData.append('effective_to', effectiveTo || '');
    formData.append('description', description.trim() || '');

    try {
      const response = await fetch(`${API_BASE}/api/policy/upload`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setPolicyStatus(data);
      setActiveTab('bill');
    } catch (error) {
      console.error('Policy upload error:', error);
      setPolicyStatus({ error: 'Failed to upload policy' });
    } finally {
      setLoading(false);
    }
  };

  const handleBillCheck = async () => {
    if (!billFile || !companyName.trim()) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', billFile);
    formData.append('company', companyName.trim());

    try {
      const response = await fetch(`${API_BASE}/api/bill/check`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setResult(data);
      setActiveTab('results');
    } catch (error) {
      console.error('Bill check error:', error);
      setResult({ error: 'Failed to check bill' });
    } finally {
      setLoading(false);
    }
  };

  const getSeverityColor = (severity) => {
    const colors = {
      HIGH: 'bg-red-100 text-red-800 border-red-300',
      MEDIUM: 'bg-yellow-100 text-yellow-800 border-yellow-300',
      LOW: 'bg-blue-100 text-blue-800 border-blue-300',
    };
    return colors[severity] || 'bg-gray-100 text-gray-800 border-gray-300';
  };

  const getScoreColor = (score) => {
    if (score >= 80) return 'text-green-600';
    if (score >= 50) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8 max-w-6xl">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Policy Compliance Checker
          </h1>
          <p className="text-gray-600">
            RAG-powered bill verification against company policies
          </p>
        </div>

        <div className="bg-white rounded-lg shadow-lg mb-6">
          {/* Tabs */}
          <div className="flex border-b">
            {['policy', 'bill', 'results'].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`flex-1 py-4 px-6 font-semibold transition-colors ${
                  activeTab === tab
                    ? 'bg-indigo-600 text-white'
                    : 'text-gray-600 hover:bg-gray-50'
                }`}
              >
                {tab === 'policy' && <FileText className="inline-block mr-2 w-5 h-5" />}
                {tab === 'bill' && <Upload className="inline-block mr-2 w-5 h-5" />}
                {tab === 'results' && <CheckCircle className="inline-block mr-2 w-5 h-5" />}
                {tab === 'policy'
                  ? 'Upload Policy'
                  : tab === 'bill'
                  ? 'Check Bill'
                  : 'Results'}
              </button>
            ))}
          </div>

          <div className="p-8">
            {/* === POLICY UPLOAD TAB === */}
            {activeTab === 'policy' && (
              <div>
                <h2 className="text-2xl font-bold text-gray-900 mb-4">
                  Upload Company Policy
                </h2>
                <p className="text-gray-600 mb-6">
                  Upload your company's policy document (any format: PDF, Image, Excel, etc.).
                </p>

                <div className="space-y-6">
                  {/* Company Name */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Company Name
                    </label>
                    <input
                      type="text"
                      value={companyName}
                      onChange={(e) => setCompanyName(e.target.value)}
                      placeholder="Enter company name"
                      className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:ring-2 focus:ring-indigo-500"
                    />
                  </div>

                  {/* Policy Name */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Policy Name
                    </label>
                    <input
                      type="text"
                      value={policyName}
                      onChange={(e) => setPolicyName(e.target.value)}
                      placeholder="e.g., Travel & Expense Policy 2024"
                      className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:ring-2 focus:ring-indigo-500"
                    />
                  </div>

                  {/* Effective Dates */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Effective From
                      </label>
                      <input
                        type="datetime-local"
                        value={effectiveFrom}
                        onChange={(e) => setEffectiveFrom(e.target.value)}
                        className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:ring-2 focus:ring-indigo-500"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Effective To
                      </label>
                      <input
                        type="datetime-local"
                        value={effectiveTo}
                        onChange={(e) => setEffectiveTo(e.target.value)}
                        className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:ring-2 focus:ring-indigo-500"
                      />
                    </div>
                  </div>

                  {/* Description */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Description
                    </label>
                    <textarea
                      rows={3}
                      value={description}
                      onChange={(e) => setDescription(e.target.value)}
                      placeholder="Brief summary of the policy..."
                      className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:ring-2 focus:ring-indigo-500"
                    />
                  </div>

                  {/* File Upload */}
                  <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-indigo-500 transition-colors">
                    <input
                      type="file"
                      accept=".pdf,.jpg,.jpeg,.png,.xls,.xlsx,.csv"
                      onChange={(e) => setPolicyFile(e.target.files[0])}
                      className="hidden"
                      id="policy-upload"
                    />
                    <label htmlFor="policy-upload" className="cursor-pointer block">
                      <FileText className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                      <p className="text-lg font-medium text-gray-700">
                        {policyFile ? policyFile.name : 'Click to upload policy file'}
                      </p>
                      <p className="text-sm text-gray-500 mt-2">
                        Accepted formats: PDF, Image, Excel (max 10MB)
                      </p>
                    </label>
                  </div>

                  {/* Upload Button */}
                  <button
                    onClick={handlePolicyUpload}
                    disabled={
                      !policyFile || !companyName.trim() || !policyName.trim() || loading
                    }
                    className="w-full bg-indigo-600 text-white py-3 px-6 rounded-lg font-semibold hover:bg-indigo-700 disabled:bg-gray-300 flex items-center justify-center"
                  >
                    {loading ? (
                      <>
                        <Loader2 className="animate-spin mr-2 w-5 h-5" />
                        Processing...
                      </>
                    ) : (
                      'Upload & Extract Rules'
                    )}
                  </button>

                  {/* Upload Status */}
                  {policyStatus && (
                    <div
                      className={`p-4 rounded-lg ${
                        policyStatus.error ? 'bg-red-50' : 'bg-green-50'
                      }`}
                    >
                      <p
                        className={`font-medium ${
                          policyStatus.error ? 'text-red-800' : 'text-green-800'
                        }`}
                      >
                        {policyStatus.error ||
                          `Policy uploaded successfully! Extracted ${
                            policyStatus.rules_count || 0
                          } rules.`}
                      </p>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* === BILL CHECK TAB === */}
            {activeTab === 'bill' && (
              <div>
                <h2 className="text-2xl font-bold text-gray-900 mb-4">
                  Check Bill Compliance
                </h2>
                <p className="text-gray-600 mb-6">
                  Upload a bill or invoice to verify compliance against your company policy.
                </p>

                <div className="space-y-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Company Name
                    </label>
                    <input
                      type="text"
                      value={companyName}
                      onChange={(e) => setCompanyName(e.target.value)}
                      placeholder="Enter company name"
                      className="w-full border border-gray-300 rounded-lg px-4 py-2 focus:ring-2 focus:ring-indigo-500"
                    />
                  </div>

                  <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-indigo-500 transition-colors">
                    <input
                      type="file"
                      accept=".pdf,.jpg,.jpeg,.png,.xls,.xlsx"
                      onChange={(e) => setBillFile(e.target.files[0])}
                      className="hidden"
                      id="bill-upload"
                    />
                    <label htmlFor="bill-upload" className="cursor-pointer block">
                      <Upload className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                      <p className="text-lg font-medium text-gray-700">
                        {billFile ? billFile.name : 'Click to upload bill'}
                      </p>
                      <p className="text-sm text-gray-500 mt-2">
                        PDF, Image, or Excel format
                      </p>
                    </label>
                  </div>

                  <button
                    onClick={handleBillCheck}
                    disabled={!billFile || !companyName.trim() || loading}
                    className="w-full bg-indigo-600 text-white py-3 px-6 rounded-lg font-semibold hover:bg-indigo-700 disabled:bg-gray-300 flex items-center justify-center"
                  >
                    {loading ? (
                      <>
                        <Loader2 className="animate-spin mr-2 w-5 h-5" />
                        Analyzing...
                      </>
                    ) : (
                      'Check Compliance'
                    )}
                  </button>
                </div>
              </div>
            )}

            {/* === RESULTS TAB === */}
            {activeTab === 'results' && result && (
              <div>
                <h2 className="text-2xl font-bold text-gray-900 mb-6">
                  Compliance Report
                </h2>

                {result.error ? (
                  <div className="bg-red-50 border border-red-200 rounded-lg p-6">
                    <p className="text-red-800 font-medium">{result.error}</p>
                  </div>
                ) : (
                  <>
                    <div className="bg-white border-2 border-gray-200 rounded-lg p-6 mb-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <h3 className="text-lg font-semibold text-gray-700 mb-2">
                            Compliance Score
                          </h3>
                          <p className="text-gray-600">
                            {result.mismatch_count || 0} violations found
                          </p>
                        </div>
                        <div
                          className={`text-6xl font-bold ${getScoreColor(
                            result.overall_score || 0
                          )}`}
                        >
                          {result.overall_score || 0}
                        </div>
                      </div>
                    </div>

                    {result.mismatches && result.mismatches.length > 0 ? (
                      <div className="space-y-4">
                        <h3 className="text-xl font-bold text-gray-900 mb-4">
                          Violations Detected
                        </h3>
                        {result.mismatches.map((mismatch, idx) => (
                          <div
                            key={idx}
                            className={`border-2 rounded-lg p-6 ${getSeverityColor(
                              mismatch.severity
                            )}`}
                          >
                            <div className="flex items-start justify-between mb-4">
                              <div className="flex items-center">
                                <XCircle className="w-6 h-6 mr-2" />
                                <h4 className="text-lg font-semibold">
                                  {mismatch.classification}
                                </h4>
                              </div>
                              <span className="px-3 py-1 rounded-full text-sm font-semibold">
                                {mismatch.severity}
                              </span>
                            </div>

                            <p className="mb-4 font-medium">
                              {mismatch.explanation}
                            </p>

                            <div className="space-y-3 text-sm">
                              <div>
                                <p className="font-semibold mb-1">Policy Rule:</p>
                                <p className="bg-white bg-opacity-50 p-3 rounded">
                                  {mismatch.company_rule_text}
                                </p>
                              </div>
                              <div>
                                <p className="font-semibold mb-1">Bill Extract:</p>
                                <p className="bg-white bg-opacity-50 p-3 rounded">
                                  {mismatch.bill_snippet}
                                </p>
                              </div>
                              <div className="flex items-center justify-between pt-2">
                                <span className="text-xs">
                                  Confidence: {Math.round((mismatch.confidence || 0) * 100)}%
                                </span>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="bg-green-50 border-2 border-green-200 rounded-lg p-8 text-center">
                        <CheckCircle className="w-16 h-16 text-green-600 mx-auto mb-4" />
                        <h3 className="text-2xl font-bold text-green-800 mb-2">
                          Fully Compliant
                        </h3>
                        <p className="text-green-700">
                          This bill meets all policy requirements!
                        </p>
                      </div>
                    )}
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ComplianceChecker;
