import React, { useState } from 'react';
import { Loader2 } from 'lucide-react';
import ImageUpload from '../components/ImageUpload';
import AnalysisResult from '../components/Analysis';

const Analysis = () => {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleImageUpload = async (imageFile: File) => {
    setLoading(true);
    setAnalysis(null);
    
    const reader = new FileReader();
    reader.onloadend = () => {
      setSelectedImage(reader.result as string);
    };
    reader.readAsDataURL(imageFile);

    setTimeout(() => {
      setLoading(false);
      setAnalysis(`Based on the dental image analysis:
1. Your teeth appear to have slight plaque buildup
2. Recommended actions:
   - Brush teeth twice daily with fluoride toothpaste
   - Floss daily to remove plaque between teeth
   - Consider using an electric toothbrush
   - Schedule regular dental checkups
3. No immediate concerns detected, but maintain good oral hygiene`);
    }, 2000);
  };

  return (
    <div className="flex-1 bg-gray-50 py-8">
      <div className="max-w-4xl mx-auto px-4">
        <h1 className="text-4xl font-bold text-gray-800 mb-4 text-center">
          Dental Image Analysis
        </h1>
        <p className="text-lg text-gray-600 mb-8 text-center">
          Upload a photo of your teeth for instant analysis and recommendations
        </p>

        <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
          <ImageUpload onUpload={handleImageUpload} />
        </div>

        {loading && (
          <div className="flex items-center justify-center p-8">
            <Loader2 className="animate-spin h-8 w-8 text-blue-500" />
            <span className="ml-3 text-lg text-gray-600">Analyzing image...</span>
          </div>
        )}

        {selectedImage && analysis && (
          <AnalysisResult imageUrl={selectedImage} analysis={analysis} />
        )}
      </div>
    </div>
  );
};

export default Analysis;