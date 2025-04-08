import React, { useState, useEffect } from 'react';

interface StepProps {
  step: number;
}

interface AnimationControlsProps {
  step: number;
  totalSteps: number;
  autoplay: boolean;
  onStepChange: (step: number) => void;
  onAutoplayToggle: () => void;
}

const MathematicalRepresentation: React.FC<StepProps> = ({ step }) => {
  return (
    <div className="flex flex-col items-center justify-center p-4">
      <h2 className="text-2xl font-bold mb-8">Mathematical Representation of 64D Space</h2>
      <div className="w-full max-w-2xl bg-blue-50 rounded-lg p-6 shadow-md">
        <div className="text-lg mb-4">A 64-dimensional vector is simply an ordered list of 64 numbers:</div>
        <div className="font-mono text-sm bg-white p-3 rounded overflow-x-auto">
          <span className="text-blue-600">v = (</span>
          <span className="text-red-500">v₁, v₂, v₃, ..., v₆₄</span>
          <span className="text-blue-600">)</span>
        </div>
        <div className="mt-8 text-lg">Each number represents a value along one of the 64 dimensions</div>
        <div className="flex justify-center mt-8">
          <div className="relative h-40 w-40">
            <div className="absolute top-0 left-0 w-full h-full flex items-center justify-center">
              <div className="bg-blue-100 w-32 h-32 rounded-full flex items-center justify-center">
                <div className="bg-blue-200 w-24 h-24 rounded-full flex items-center justify-center">
                  <div className="bg-blue-300 w-16 h-16 rounded-full flex items-center justify-center">
                    <div className="bg-blue-400 w-8 h-8 rounded-full"></div>
                  </div>
                </div>
              </div>
            </div>
            <div className="absolute top-1/2 left-0 w-full h-1 bg-red-400 transform -translate-y-1/2"></div>
            <div className="absolute top-0 left-1/2 w-1 h-full bg-green-400 transform -translate-x-1/2"></div>
            <div 
              className="absolute top-0 left-0 w-full h-full border border-blue-500"
            ></div>
          </div>
        </div>
        <div className="text-center mt-4 italic">Just as 3D space needs 3 coordinates, 64D space needs 64 coordinates</div>
      </div>
    </div>
  );
};

const ComputerMemoryRepresentation: React.FC<StepProps> = ({ step }) => {
  return (
    <div className="flex flex-col items-center justify-center p-4">
      <h2 className="text-2xl font-bold mb-8">Computer Memory Representation</h2>
      <div className="w-full max-w-2xl bg-green-50 rounded-lg p-6 shadow-md">
        <div className="text-lg mb-4">In memory, a 64-dimensional vector is stored as a contiguous array:</div>
        <div className="font-mono text-sm bg-white p-3 rounded overflow-x-auto">
          <span className="text-green-600">float vector[64] = {'{'}</span>
          <span className="text-purple-500">0.24, -1.42, 0.35, 0.91, 0.02, -0.56, 0.78, 1.21, -0.63, 0.44, ...</span>
          <span className="text-green-600">{'}'};</span>
        </div>
      </div>
    </div>
  );
};

const NeuralNetworkRepresentation: React.FC<StepProps> = ({ step }) => {
  return (
    <div className="flex flex-col items-center justify-center p-4">
      <h2 className="text-2xl font-bold mb-8">Neural Network Representation</h2>
      <div className="w-full max-w-2xl bg-purple-50 rounded-lg p-6 shadow-md">
        <div className="text-lg mb-4">In a neural network layer with 64 channels/neurons:</div>
        <div className="flex justify-center">
          <div className="relative w-full max-w-lg h-64">
            {/* Neural network visualization */}
          </div>
        </div>
      </div>
    </div>
  );
};

const VisualizingHigherDimensions: React.FC<StepProps> = ({ step }) => {
  return (
    <div className="flex flex-col items-center justify-center p-4">
      <h2 className="text-2xl font-bold mb-8">Visualizing Higher Dimensions</h2>
      <div className="w-full max-w-2xl bg-yellow-50 rounded-lg p-6 shadow-md">
        <div className="text-lg mb-4">We can use dimensionality reduction to project 64D space to 2D:</div>
        <div className="flex justify-center">
          <div className="relative w-full max-w-lg h-64">
            {/* Dimensionality reduction visualization */}
          </div>
        </div>
      </div>
    </div>
  );
};

const FeatureVisualization: React.FC<StepProps> = ({ step }) => {
  return (
    <div className="flex flex-col items-center justify-center p-4">
      <h2 className="text-2xl font-bold mb-8">Feature Visualization</h2>
      <div className="w-full max-w-2xl bg-red-50 rounded-lg p-6 shadow-md">
        <div className="text-lg mb-4">We can visualize what each dimension "responds to" in CNN feature maps:</div>
        <div className="flex justify-center">
          <div className="relative w-full max-w-lg h-64">
            {/* Feature visualization */}
          </div>
        </div>
      </div>
    </div>
  );
};

const PracticalCNNExample: React.FC<StepProps> = ({ step }) => {
  return (
    <div className="flex flex-col items-center justify-center p-4">
      <h2 className="text-2xl font-bold mb-8">Practical CNN Example</h2>
      <div className="w-full max-w-2xl bg-indigo-50 rounded-lg p-6 shadow-md">
        <div className="text-lg mb-4">In a U-Net or similar CNN architecture:</div>
        <div className="flex justify-center">
          <div className="relative w-full max-w-lg h-64">
            {/* CNN architecture visualization */}
          </div>
        </div>
      </div>
    </div>
  );
};

const AnimationControls: React.FC<AnimationControlsProps> = ({
  step,
  totalSteps,
  autoplay,
  onStepChange,
  onAutoplayToggle
}) => {
  const handlePrevious = () => {
    onStepChange((step - 1 + totalSteps) % totalSteps);
  };

  const handleNext = () => {
    onStepChange((step + 1) % totalSteps);
  };

  return (
    <div className="p-4 bg-gray-100 flex justify-between items-center">
      <button 
        onClick={handlePrevious}
        className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50"
        aria-label="Previous step"
      >
        Previous
      </button>
      
      <div className="flex items-center">
        <span className="mr-4" aria-live="polite">Step {step + 1} of {totalSteps}</span>
        <button 
          onClick={onAutoplayToggle}
          className={`px-4 py-2 rounded focus:outline-none focus:ring-2 focus:ring-opacity-50 ${
            autoplay ? 'bg-red-500 text-white focus:ring-red-500' : 'bg-green-500 text-white focus:ring-green-500'
          }`}
          aria-label={autoplay ? 'Pause animation' : 'Start animation'}
        >
          {autoplay ? 'Pause' : 'Autoplay'}
        </button>
      </div>
      
      <button 
        onClick={handleNext}
        className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50"
        aria-label="Next step"
      >
        Next
      </button>
    </div>
  );
};

const DimensionalSpaceAnimation: React.FC = () => {
  const [step, setStep] = useState<number>(0);
  const [autoplay, setAutoplay] = useState<boolean>(false);
  const totalSteps = 6;
  const ANIMATION_INTERVAL = 3500;
  
  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (autoplay) {
      timer = setInterval(() => {
        setStep((prevStep) => (prevStep + 1) % totalSteps);
      }, ANIMATION_INTERVAL);
    }
    return () => clearInterval(timer);
  }, [autoplay]);

  const renderStep = () => {
    switch(step) {
      case 0:
        return <MathematicalRepresentation step={step} />;
      case 1:
        return <ComputerMemoryRepresentation step={step} />;
      case 2:
        return <NeuralNetworkRepresentation step={step} />;
      case 3:
        return <VisualizingHigherDimensions step={step} />;
      case 4:
        return <FeatureVisualization step={step} />;
      case 5:
        return <PracticalCNNExample step={step} />;
      default:
        return null;
    }
  };

  const handleAutoplayToggle = () => {
    setAutoplay(!autoplay);
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <header className="p-4 bg-gray-800 text-white">
        <h1 className="text-2xl font-bold text-center">Understanding 64-Dimensional Space in Neural Networks</h1>
      </header>
      
      <main className="flex-grow flex flex-col items-center justify-center p-4">
        {renderStep()}
      </main>
      
      <AnimationControls
        step={step}
        totalSteps={totalSteps}
        autoplay={autoplay}
        onStepChange={setStep}
        onAutoplayToggle={handleAutoplayToggle}
      />
    </div>
  );
};

export default DimensionalSpaceAnimation; 