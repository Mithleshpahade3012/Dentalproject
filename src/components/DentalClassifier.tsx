"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Button } from "../components/ui/button";
import { Card, CardContent } from "../components/ui/card";
import '../styles/main.css'
// import Navbar from "./Navbar";

export default function DentalClassifier() {
  const [image, setImage] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [prediction, setPrediction] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const uploadedFile = event.target.files?.[0];
    if (uploadedFile) {
      setFile(uploadedFile);
      setImage(URL.createObjectURL(uploadedFile));
    }
  };

  const handlePredict = async () => {
    if (!file) return alert("Please upload an image first.");
    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://192.168.1.45:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Failed to fetch prediction");

      const data = await response.json();
      setPrediction(data);
    } catch (error) {
      console.error("Error:", error);
      alert("Prediction failed. Check the server.");
    } finally {
      setLoading(false);
    }
  };

  const handleReupload = () => {
    setImage(null);
    setFile(null);
    setPrediction(null);
  };

  return (
    <div className="min-h-screen bg-[url('/download.jpg')] bg-cover bg-center">
      <div className="bg-image"></div>

      {/* Navbar */}
      {/* <Navbar /> */}
      <div className="max-w-4xl mx-auto px-4" style={{
        padding : "100px 0px 0px 0px"
      }}>
        <h1 className="text-4xl font-bold text-gray-800 mb-4 text-center">
          Dental Image Analysis
        </h1>
        <p className="text-lg text-gray-600 mb-8 text-center">
          Upload a photo of your teeth for instant analysis and recommendations
        </p>
      </div>
        <div className="flex flex-col md:flex-row items-center justify-center  p-6 gap-6">
            <Card className="p-6 w-full max-w-md bg-white shadow-lg rounded-xl">
            <CardContent className="flex flex-col items-center">
                <motion.h1 
                className="text-2xl font-bold mb-4" 
                animate={{ opacity: 1, y: 0 }} 
                initial={{ opacity: 0, y: -20 }}
                >
                
                </motion.h1>

                <label
                htmlFor="file-upload"
                className="cursor-pointer border-2 border-dashed border-gray-400 rounded-lg p-4 w-full text-center hover:border-blue-500 transition"
                >
                <p className="text-gray-600">Click to Upload Image</p>
                <input id="file-upload" type="file" accept="image/*" onChange={handleImageUpload} className="hidden" />
                </label>
                {image && <img src={image} alt="Uploaded" className="rounded-lg mt-4 mb-4 shadow-md w-[200px]" />}
                <br></br>
                <Button onClick={handlePredict} disabled={loading} className="w-full bg-blue-700 hover:bg-blue-800">
                {loading ? "Analyzing..." : "🦷 Predict Condition"}
                </Button>
            </CardContent>
            </Card>

            {prediction && (
            <motion.div 
                className="w-full max-w-lg bg-white shadow-lg rounded-xl p-6 flex flex-col items-center"
                initial={{ opacity: 0, x: 50 }}
                animate={{ opacity: 1, x: 0 }}
            >  
                {prediction.gradcam_base64 && (
                <div className="flex-shrink-0 text-center">
                    <h2 className="text-xl font-bold mb-2">🩺 Prediction</h2>
                    <img
                    src={`data:image/png;base64,${prediction.gradcam_base64}`}
                    alt="Grad-CAM"
                    className="rounded-lg shadow-md w-[300px]"
                    />
                    <p className="text-sm text-blue-600 mt-1"></p>
                </div>
                )}

                <div className="text-center mt-4">
                <p className="text-3xl font-bold mb-2">Disease: <span className="text-grey-900">{prediction.predicted_disease}</span></p>
                <p className="text-3xl font-bold mb-2">Condition: <span className="text-gray-900">{prediction.condition}</span></p>
                <p className="text-2xl  font-bold text-gray-600 mb-2">Prediction Accuracy: <span className="text-green-500">{prediction.confidence}</span></p>
                <p className="text-lg text-gray-600 mb-2">Advice: <span className="text-gray-700">{prediction.advice}</span></p>
                <Button onClick={handleReupload} className="mt-4 w-full text-black bg-red-50 hover:bg-red-500">
                    Close
                </Button>
                </div>
            </motion.div>
            )}
        </div>
    </div>
  );
}
