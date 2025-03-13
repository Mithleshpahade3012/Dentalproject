import React from 'react';
import { ArrowRight, Shield, Clock, Award } from 'lucide-react';
import { Link } from 'react-router-dom';

const Home = () => {
  return (
    <div className="flex-1">
      <div
        className="h-[500px] bg-cover bg-center relative"
        style={{
          backgroundImage:
            'url(https://images.unsplash.com/photo-1606811841689-23dfddce3e95?q=80&w=1920&auto=format&fit=crop)',
        }}
      >
        <div className="absolute inset-0 bg-white/70 flex items-center justify-center">
          <div className="text-center text-gray-800 px-4">
            <h1 className="text-5xl font-bold mb-6">AI-Powered Dental Analysis</h1>
            <p className="text-xl mb-8 max-w-2xl">
              Get instant insights about your oral health using our advanced AI
              technology. Upload a photo and receive personalized recommendations.
            </p>
            <Link
              to="/analysis"
              className="inline-flex items-center bg-blue-600 text-white px-8 py-3 rounded-full text-lg font-semibold hover:bg-blue-700 transition-colors"
            >
              Try Analysis Now
              <ArrowRight className="ml-2 h-5 w-5" />
            </Link>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-16">
        <div className="grid md:grid-cols-3 gap-8">
          <div className="bg-white rounded-xl shadow-lg p-6 text-center">
            <Shield className="h-12 w-12 text-blue-600 mx-auto mb-4" />
            <h3 className="text-xl font-semibold mb-2 text-gray-800">Safe & Secure</h3>
            <p className="text-gray-600">
              Your dental images are processed securely and never stored without
              permission.
            </p>
          </div>
          <div className="bg-white rounded-xl shadow-lg p-6 text-center">
            <Clock className="h-12 w-12 text-blue-600 mx-auto mb-4" />
            <h3 className="text-xl font-semibold mb-2 text-gray-800">Instant Results</h3>
            <p className="text-gray-600">
              Get immediate analysis and recommendations for your oral health.
            </p>
          </div>
          <div className="bg-white rounded-xl shadow-lg p-6 text-center">
            <Award className="h-12 w-12 text-blue-600 mx-auto mb-4" />
            <h3 className="text-xl font-semibold mb-2 text-gray-800">Expert System</h3>
            <p className="text-gray-600">
              Powered by advanced AI trained on thousands of dental images.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;