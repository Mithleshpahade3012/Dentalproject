import React from 'react';

interface AnalysisProps {
  imageUrl: string;
  analysis: string;
}

const Analysis: React.FC<AnalysisProps> = ({ imageUrl, analysis }) => {
  return (
    <div className="bg-white rounded-xl shadow-lg overflow-hidden">
      <div className="md:flex">
        <div className="md:w-1/2">
          <img
            src={imageUrl}
            alt="Uploaded dental image"
            className="w-full h-full object-cover"
          />
        </div>
        <div className="md:w-1/2 p-6">
          <h3 className="text-2xl font-semibold text-gray-900 mb-4">Analysis Results</h3>
          <div className="prose">
            {analysis.split('\n').map((line, index) => (
              <p key={index} className="mb-2 text-gray-700">
                {line}
              </p>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Analysis;